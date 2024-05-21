import argparse
import torch
import numpy as np
import math
import wandb
from src.data.KITTI_dataset import KITTI, SequenceBoundarySampler
from src.data.KITTI_eval import KITTI_tester
from src.models.DeepVIO import DeepVIO
from src.models.DeepVIO_CDE import DeepVIO_CDE
from utils.params import set_gpu_ids, load_pretrained_model, get_optimizer
from utils.utils import setup_experiment_directories, setup_training_logger, setup_debug_logger, print_tensor_stats
from scripts.transforms import get_transforms


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Paths for data and results
parser.add_argument( "--data_dir", type=str, default="/mnt/data0/marco/KITTI/data", help="path to the dataset",)
parser.add_argument( "--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU",)
parser.add_argument( "--save_dir", type=str, default="./results", help="path to save the result")
parser.add_argument( "--plot_dir", type=str, default="./results", help="path to save the log")

# Training Configurations
parser.add_argument( "--experiment_name", type=str, default="experiment", help="experiment name")
parser.add_argument( "--wandb", default=False, action="store_true", help="whether to use wandb logging")
parser.add_argument( "--resume", type=str, default=None, help="resume training (wandb run id)")
parser.add_argument( "--pretrain_flownet", type=str, default="./pretrained_models/flownets_bn_EPE2.459.pth.tar", help="wehther to use the pre-trained flownet",)
parser.add_argument( "--pretrain", type=str, default=None, help="path to the pretrained model")
parser.add_argument( "--train_seq", type=str, default=["04", "10"], nargs="+", help="sequences for training",)
parser.add_argument( "--val_seq", type=str, default=["04", "10"], nargs="+", help="sequences for validation",)
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--workers", type=int, default=8, help="number of workers in dataloader")
parser.add_argument( "--print_frequency", type=int, default=10, help="print frequency for loss values")

# Training Hyperparameters
parser.add_argument( "--model_type", type=str, default="ode-rnn", help="type of model [rnn, ode, ode-rnn, ltc, cfc]")
parser.add_argument( "--optimizer", type=str, default="Adam", help="type of optimizer [Adam, SGD]")
parser.add_argument( "--grad_accumulation_steps", type=int, default=1, help="gradient accumulation steps before updating")
parser.add_argument( "--freeze_encoder", default=False, action="store_true", help="freeze the encoder or not")
parser.add_argument( "--weight_decay", type=float, default=5e-6, help="weight decay for the optimizer")
parser.add_argument("--batch_size", type=int, default=26, help="batch size")
parser.add_argument( "--shuffle", type=int, default=True, help="shuffle data samples or not")
parser.add_argument( "--epochs_warmup", type=int, default=20, help="number of epochs for warmup")
parser.add_argument( "--epochs_joint", type=int, default=40, help="number of epochs for joint training")
parser.add_argument( "--epochs_fine", type=int, default=40, help="number of epochs for finetuning")
parser.add_argument( "--lr_warmup", type=float, default=1e-4, help="learning rate for warming up stage")
parser.add_argument( "--lr_joint", type=float, default=1e-5, help="learning rate for joint training stage",)
parser.add_argument( "--lr_fine", type=float, default=1e-6, help="learning rate for finetuning stage")
parser.add_argument( "--gradient_clip", type=float, default=5, help="gradient clipping norm/clip value")

# Data Augmentation
parser.add_argument("--data_dropout", type=float, default=0.0, help="irregularity in the dataset by dropping out randomly")
parser.add_argument("--eval_data_dropout", type=float, default=0.0, help="irregularity in the eval dataset")
parser.add_argument("--img_w", type=int, default=512, help="image width")
parser.add_argument("--img_h", type=int, default=256, help="image height")
parser.add_argument("--v_f_len", type=int, default=512, help="visual feature length")
parser.add_argument("--i_f_len", type=int, default=256, help="imu feature length")
parser.add_argument( "--imu_dropout", type=float, default=0, help="dropout for the IMU encoder")
parser.add_argument( "--hflip", default=False, action="store_true", help="whether to use horizonal flipping as augmentation",)
parser.add_argument( "--color", default=False, action="store_true", help="whether to use color augmentations",)
parser.add_argument("--seq_len", type=int, default=11, help="sequence length of images")

# Fusion Module Parameters
parser.add_argument( "--fuse_method", type=str, default="cat", help="fusion method of encoded IMU and Images[cat, soft, hard]")

# ODE Parameters
parser.add_argument( "--ode_hidden_dim", type=int, default=512, help="size of the ODE latent")
parser.add_argument( "--ode_num_layers", type=int, default=3, help="number of layers for the ODE")
parser.add_argument( "--ode_activation_fn", type=str, default="tanh", help="activation function [softplus, relu, leaky_relu, tanh]",)
parser.add_argument( "--ode_solver", type=str, default="dopri5", help="ODE solvers [dopri5, heun, euler, runge_kutta, tsit5]",)

# RNN Regressor Parameters for ODE-RNN/NCP Implementation
parser.add_argument( "--ode_rnn_type", type=str, default="rnn", help="type of RNN [rnn, lstm, gru]") 
parser.add_argument( "--rnn_num_layers", type=int, default=2, help="number of layers for RNN")
parser.add_argument( "--rnn_hidden_size", type=int, default=1024, help="size of the RNN latent") 
parser.add_argument( "--rnn_dropout_out", type=float, default=0, help="dropout for the RNN output layer",)

# CDE Parameters 
parser.add_argument( "--cde_hidden_dim", type=int, default=128, help="size of the CDE latent")
parser.add_argument( "--cde_num_layers", type=int, default=3, help="number of layers for the cDE")
parser.add_argument( "--cde_activation_fn", type=str, default="tanh", help="activation function [softplus, relu, leaky_relu, tanh]",)
parser.add_argument( "--adjoint", default=False, action="store_true", help="whether to use adjoint method",)

args = parser.parse_args()



# Set the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Setup experiment directories and logger
checkpoints_dir, log_dir, graph_dir, flownet_dir, img_dir = setup_experiment_directories(args)
logger = setup_training_logger(args, log_dir)
debug_logger = setup_debug_logger(args, log_dir)

def update_status(ep, args, model):
    if ep < args.epochs_warmup:  # Warmup stage
        lr = args.lr_warmup
    elif (
        ep >= args.epochs_warmup and ep < args.epochs_warmup + args.epochs_joint
    ):  # Joint training stage
        lr = args.lr_joint
    elif ep >= args.epochs_warmup + args.epochs_joint:  # Finetuning stage
        lr = args.lr_fine
    return lr

def train(model, optimizer, train_loader, logger, ep):
    mse_losses = []
    data_len = len(train_loader)
    optimizer.zero_grad()
    for i, (imgs, imus, gts, timestamps, folder) in enumerate(
        train_loader
    ):
        # imgs.shape, imus.shape = torch.Size([batch_size, 11, 3, 256, 512]), torch.Size([batch_size, 101, 6])
        # Reason why imus has 101 samples is becuase there there are 10 samples per 1 image, between 2 images, there are 11 imu samples. So between 11 images there are 100 samples. The last image also has 1 imu data, thus 101 samples including boundary of 11 images.
        imgs = imgs.cuda().float()
        imus = imus.cuda().float()
        gts = gts.cuda().float()
        timestamps = timestamps.cuda().float()

        # imgs.shape, imus.shape, timestamps.shape = torch.Size([32, 11, 3, 256, 512]) torch.Size([32, 101, 6]) torch.Size([32, 11])
        poses, _ = model(
            imgs,
            imus,
            timestamps,
            is_first=True,
            hc=None,
        )
        
        # poses shape = torch.Size([batch_size, 10, 6])
        # Calculate relative pose change from one prediction befores
        relative_pose = poses

        # Calculate angle and translation loss
        angle_loss = torch.nn.functional.mse_loss(
            relative_pose[:, :, :3], gts[:, :, :3]
        )
        translation_loss = torch.nn.functional.mse_loss(
            relative_pose[:, :, 3:], gts[:, :, 3:]
        )

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
            
        if i % args.print_frequency == 0:
            message = f"Epoch: {ep}, iters: {i}/{data_len}, pose loss: {pose_loss.item():.6f}, angle_loss: {angle_loss.item():.6f}, translation_loss: {translation_loss.item():.6f}, loss: {loss.item():.6f}"
            logger.info(message)
        mse_losses.append(pose_loss.item())
    return np.mean(mse_losses)


def evaluate(model, tester, ep, best):
    # Evaluate the model
    logger.info("Evaluating the model")
    with torch.no_grad():
        model.eval()
        errors = tester.eval(model, num_gpu=1)
        tester.generate_plots(graph_dir, ep)

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


def main():
    # Get the data transforms
    transform_train = get_transforms(args)

    # Load the dataset
    train_dataset = KITTI(
        args.data_dir,
        sequence_length=args.seq_len,
        train_seqs=args.train_seq,
        transform=transform_train,
        logger=debug_logger,
        dropout=args.data_dropout,
    )

    # Using batch sampler here to preserve the sequence boundaries, preventing non-ascending timestamps
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
    gpu_id = set_gpu_ids(args)

    # Model initialization
    model = DeepVIO_CDE(args)

    # Continual training or not
    load_pretrained_model(model, args)

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

    if args.freeze_encoder:
        for param in model.Image_net.parameters():
            param.requires_grad = False
        logger.info("Frozen encoder")
            
    # Initialize the tester
    tester = KITTI_tester(args)

    # Feed model to GPU
    model.cuda(gpu_id)
    model = torch.nn.DataParallel(model, device_ids=[gpu_id])

    pretrain = args.pretrain
    init_epoch = int(pretrain[-7:-4]) + 1 if args.pretrain is not None else 0

    # Initialize the optimizer
    optimizer = get_optimizer(model, args)
    best = 10000

    for ep in range(
        init_epoch, args.epochs_warmup + args.epochs_joint + args.epochs_fine
    ):
        lr = update_status(ep, args, model)
        optimizer.param_groups[0]["lr"] = lr
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
        logger.info(f"Wandb Run ID: {id}")
        wandb.init(
            # set the wandb project where this run will be logged
            project="Final Year Project",
            id=id, 
            resume=resume,
            name=args.experiment_name,
            # track hyperparameters and run metadata
            config=args,
        )
    main()
