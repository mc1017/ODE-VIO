import torch
import numpy as np
import wandb
from src.data.KITTI_dataset import KITTI, SequenceBoundarySampler
from src.data.KITTI_eval import KITTI_tester
from src.models.DeepVIO import DeepVIO
from utils.utils import setup_experiment_directories, setup_training_logger, setup_debug_logger, print_tensor_stats, set_gpu_ids, load_pretrained_model, get_optimizer
from utils.profiler import trace_handler, log_parameter_count
from torch.autograd.profiler import record_function
from src.data.transforms import get_transforms
from scripts.config import get_args
from torch.profiler import profile, record_function, ProfilerActivity

args = get_args()

# Set the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Setup experiment directories and logger
checkpoints_dir, log_dir, graph_dir, flownet_dir, img_dir = setup_experiment_directories(args)
logger = setup_training_logger(args, log_dir)
debug_logger = setup_debug_logger(args, log_dir)

def update_status(ep, args, model):
    if ep < args.epochs_warmup:  # Warmup stage
        # lr = get_warmup_lr(ep, args.epochs_warmup, args.lr_warmup, args.lr_joint)
        lr = args.lr_warmup
    elif (
        ep >= args.epochs_warmup and ep < args.epochs_warmup + args.epochs_joint
    ):  # Joint training stage
        lr = args.lr_joint
    elif ep >= args.epochs_warmup + args.epochs_joint:  # Finetuning stage
        lr = args.lr_fine
    return lr

def get_warmup_lr(epoch, warmup_epochs, base_lr, warmup_lr):
    if epoch < warmup_epochs:
        lr = warmup_lr + (base_lr - warmup_lr) * (epoch / warmup_epochs)
    else:
        lr = base_lr
    return lr

def profile_memory():
    print(torch.cuda.memory_summary())
    torch.cuda.empty_cache()

def train(model, optimizer, train_loader, logger, ep):
    """
    Main Training Loop for the model
    """
    mse_losses = []
    data_len = len(train_loader)
    optimizer.zero_grad()
    # torch.cuda.memory._record_memory_history(max_entries=10000000) # For memory profiling
    
    for i, (imgs, imus, gts, timestamps, folder) in enumerate(
        train_loader
    ):  
            torch.cuda.empty_cache()
            
            # Reason why imus has 101 samples is becuase there there are 10 samples per 1 image, between 2 images, there are 11 imu samples. So between 11 images there are 100 samples. The last image also has 1 imu data, thus 101 samples including boundary of 11 images.
            imgs = imgs.cuda().float()
            imus = imus.cuda().float()
            gts = gts.cuda().float()
            timestamps = timestamps.cuda().float()

            # imgs.shape, imus.shape, timestamps.shape = torch.Size([32, 11, 3, 256, 512]) torch.Size([32, 101, 6]) torch.Size([32, 11])
            poses,_ = model(imgs, imus, timestamps, hc=None)
            
            # Calculate angle and translation loss
            angle_loss = torch.nn.functional.mse_loss( poses[:, :, :3], gts[:, :, :3])
            translation_loss = torch.nn.functional.mse_loss( poses[:, :, 3:], gts[:, :, 3:])

            # Calculate Loss
            pose_loss = 100 * angle_loss + translation_loss
            loss = pose_loss
            loss.backward()
            
            
            # Gradient accumulation
            if (i + 1) % args.grad_accumulation_steps == 0 or (i + 1) == data_len:
                if args.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip)
                    optimizer.step()
                optimizer.zero_grad()
                
                
            if (i+1) % args.print_frequency == 0:
                message = f"Epoch: {ep}, iters: {i+1}/{data_len}, pose loss: {pose_loss.item():.6f}, angle_loss: {angle_loss.item():.6f}, translation_loss: {translation_loss.item():.6f}, loss: {loss.item():.6f}"
                logger.info(message)
                
            mse_losses.append(pose_loss.item())
    # torch.cuda.memory._record_memory_history(enabled=None) # (For Memory Profiling)
    return np.mean(mse_losses)


def evaluate(model, tester, ep, best):
    """
    Evaluate the model on the evaluation dataset. It saves the model, calculates all parameters, and generate a plot
    """
    logger.info("Evaluating the model")
    with torch.no_grad():
        model.eval()
        errors = tester.eval(model, num_gpu=1)
        tester.generate_plots(graph_dir, ep) # Optional: Generate plots for the evaluation

    t_rel = np.mean([errors[i]["t_rel"] for i in range(len(errors))])
    r_rel = np.mean([errors[i]["r_rel"] for i in range(len(errors))])
    t_rmse = np.mean([errors[i]["t_rmse"] for i in range(len(errors))])
    r_rmse = np.mean([errors[i]["r_rmse"] for i in range(len(errors))])
    
    if t_rel < best:
        best = t_rel
        torch.save(model.module.state_dict(), f"{checkpoints_dir}/best_{best:.2f}.pth")

    message = f"Epoch {ep} evaluation finished , t_rel: {t_rel:.4f}, r_rel: {r_rel:.4f}, t_rmse: {t_rmse:.4f}, r_rmse: {r_rmse:.4f}, best t_rel: {best:.4f}"
    logger.info(message)
    return best, t_rel, r_rel, t_rmse, r_rmse


def get_train_loader(args):
    # Get the data transforms
    transform_train = get_transforms(args)
    dropout_ratio = np.random.normal(args.data_dropout, args.data_dropout_std) 
    logger.info("Dropout ratio: {}".format(dropout_ratio))
    
    # Load the dataset
    train_dataset = KITTI(
        args.data_dir,
        sequence_length=args.seq_len,
        train_seqs=args.train_seq,
        transform=transform_train,
        logger=debug_logger,
        dropout=dropout_ratio,
    )
    batch_sampler = SequenceBoundarySampler(
        args.data_dir,
        batch_size=args.batch_size,
        train_seqs=args.train_seq,
        seq_len=args.seq_len,
        shuffle=args.shuffle,
        img_seq_length=train_dataset.img_seq_len,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=args.workers,
        pin_memory=True,
        batch_sampler=batch_sampler,
    )
    return train_loader

def load_pretrain_flownet(model, args):
    pretrained_w = torch.load(args.pretrain_flownet, map_location="cpu")
    model_dict = model.Image_net.state_dict()
    update_dict = {
        k: v for k, v in pretrained_w["state_dict"].items() if k in model_dict
    }
    model_dict.update(update_dict)
    model.Image_net.load_state_dict(model_dict)
    logger.info("Pretrained flownet loaded")

def main():
    """
    Main entry point
    """
    
    gpu_id = set_gpu_ids(args)

    # Model initialization
    model = DeepVIO(args)
    log_parameter_count(model, logger)
    
    # Continual training or not
    load_pretrained_model(model, args)
    pretrain = args.pretrain
    init_epoch = int(pretrain[-7:-4]) + 1 if args.pretrain is not None else 0

    # Use the pre-trained flownet or not
    if args.pretrain_flownet and args.pretrain is None:
        pretrained_w = torch.load(args.pretrain_flownet, map_location="cpu")
        model_dict = model.Image_net.state_dict()
        update_dict = {
            k: v for k, v in pretrained_w["state_dict"].items() if k in model_dict
        }
        model_dict.update(update_dict)
        model.Image_net.load_state_dict(model_dict)
        logger.info("Pretrained flownet loaded")

    # Freeze the encoder
    if args.freeze_encoder:
        for param in model.Image_net.parameters():
            param.requires_grad = False
        logger.info("Frozen encoder")
            
    # Initialize the tester
    tester = KITTI_tester(args)

    # Feed model to GPU
    model.cuda(gpu_id)
    model = torch.nn.DataParallel(model, device_ids=[gpu_id])

    # Initialize the optimizer
    optimizer = get_optimizer(model, args)
    best = 10000
    
    for ep in range(
        init_epoch, args.epochs_warmup + args.epochs_joint + args.epochs_fine
    ):
        train_loader = get_train_loader(args)
        
        lr = update_status(ep, args, model)
        
        # Create parameter groups for layer based learning rates (Optional)
        optimizer.param_groups[0]["lr"] = lr
        # optimizer.param_groups[1]["lr"] = lr
        message = f"Epoch: {ep}, lr: {lr}"
        logger.info(message)
        model.train()
        avg_pose_loss = train(model, optimizer, train_loader, logger, ep)

        # Save the model after training
        if ep % 2 == 0:
            torch.save(model.module.state_dict(), f"{checkpoints_dir}/{ep:003}.pth")
        message = f"Epoch {ep} training finished, pose loss: {avg_pose_loss:.6f}"
        logger.info(message)
        best, t_rel, r_rel, t_rmse, r_rmse  = evaluate(model, tester, ep, best)
        if args.wandb:
            wandb.log({"t_rel": t_rel, "r_rel": r_rel, "t_rmse": t_rmse, "r_rmse": r_rmse, "best_t_rel": best, "avg_pose_loss": avg_pose_loss})
        
    message = f"Training finished, best t_rel: {best:.4f}"
    logger.info(message)



if __name__ == "__main__":
    if args.wandb:
        id, resume = (args.resume, "must") if args.resume else (wandb.util.generate_id(), "allow")
        group = args.wandb_group
        logger.info(f"Wandb Run ID: {id}")
        wandb.init(
            project="Final Year Project",
            group=group,
            id=id, 
            resume=resume,
            name=args.experiment_name,
            config=args,
        )
    main()
   
    
