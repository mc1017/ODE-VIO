import argparse
import torch
import numpy as np
from pathlib import Path
import math
import inspect
from src.data.KITTI_dataset import KITTI, SequenceBoundarySampler
from src.data.utils import *
from src.data.KITTI_eval import KITTI_tester
from src.models.NeuralODE import DeepVIO
from utils.params import set_gpu_ids, load_pretrained_model, get_optimizer
from utils.utils import setup_experiment_directories, setup_logger


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='/mnt/data0/marco/KITTI/data', help='path to the dataset')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')

parser.add_argument('--train_seq', type=list, default=['00', '01', '02', '04', '06', '08', '09'], help='sequences for training')
parser.add_argument('--val_seq', type=list, default=['05', '07', '10'], help='sequences for validation')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--img_w', type=int, default=512, help='image width')
parser.add_argument('--img_h', type=int, default=256, help='image height')
parser.add_argument('--v_f_len', type=int, default=512, help='visual feature length')
parser.add_argument('--i_f_len', type=int, default=256, help='imu feature length')
parser.add_argument('--fuse_method', type=str, default='cat', help='fusion method [cat, soft, hard]')
parser.add_argument('--imu_dropout', type=float, default=0, help='dropout for the IMU encoder')

parser.add_argument('--rnn_hidden_size', type=int, default=1024, help='size of the LSTM latent')
parser.add_argument('--rnn_dropout_out', type=float, default=0.2, help='dropout for the LSTM output layer')
parser.add_argument('--rnn_dropout_between', type=float, default=0.2, help='dropout within LSTM')

parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for the optimizer')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--seq_len', type=int, default=11, help='sequence length for LSTM')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--epochs_warmup', type=int, default=40, help='number of epochs for warmup')
parser.add_argument('--epochs_joint', type=int, default=40, help='number of epochs for joint training')
parser.add_argument('--epochs_fine', type=int, default=20, help='number of epochs for finetuning')
parser.add_argument('--lr_warmup', type=float, default=5e-4, help='learning rate for warming up stage')
parser.add_argument('--lr_joint', type=float, default=5e-5, help='learning rate for joint training stage')
parser.add_argument('--lr_fine', type=float, default=1e-6, help='learning rate for finetuning stage')
parser.add_argument('--eta', type=float, default=0.05, help='exponential decay factor for temperature')
parser.add_argument('--temp_init', type=float, default=5, help='initial temperature for gumbel-softmax')
parser.add_argument('--Lambda', type=float, default=3e-5, help='penalty factor for the visual encoder usage')

parser.add_argument('--experiment_name', type=str, default='experiment', help='experiment name')
parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer [Adam, SGD]')

parser.add_argument('--pretrain_flownet',type=str, default='./pretrained_models/flownets_bn_EPE2.459.pth.tar', help='wehther to use the pre-trained flownet')
parser.add_argument('--pretrain', type=str, default=None, help='path to the pretrained model')
parser.add_argument('--hflip', default=False, action='store_true', help='whether to use horizonal flipping as augmentation')
parser.add_argument('--color', default=False, action='store_true', help='whether to use color augmentations')

parser.add_argument('--print_frequency', type=int, default=10, help='print frequency for loss values')
parser.add_argument('--weighted', default=False, action='store_true', help='whether to use weighted sum')
args = parser.parse_args()


# Set the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

 # Setup experiment directories and logger
checkpoints_dir, log_dir = setup_experiment_directories(args)
logger = setup_logger(args, log_dir)


def update_status(ep, args, model):
    if ep < args.epochs_warmup:  # Warmup stage
        lr = args.lr_warmup
        selection = 'random'
        temp = args.temp_init
        for param in model.module.Policy_net.parameters(): # Disable the policy network
            param.requires_grad = False
    elif ep >= args.epochs_warmup and ep < args.epochs_warmup + args.epochs_joint: # Joint training stage
        lr = args.lr_joint
        selection = 'gumbel-softmax'
        temp = args.temp_init * math.exp(-args.eta * (ep-args.epochs_warmup))
        for param in model.module.Policy_net.parameters(): # Enable the policy network
            param.requires_grad = True
    elif ep >= args.epochs_warmup + args.epochs_joint: # Finetuning stage
        lr = args.lr_fine
        selection = 'gumbel-softmax'
        temp = args.temp_init * math.exp(-args.eta * (ep-args.epochs_warmup))
    return lr, selection, temp

def train(model, optimizer, train_loader, selection, temp, logger, ep, p=0.5, weighted=False):
    
    mse_losses = []
    penalties = [0]
    data_len = len(train_loader)
    
    for i, (imgs, imus, gts, rot, weight, timestamps) in enumerate(train_loader):
       
        # imgs.shape, imus.shape = torch.Size([16, 11, 3, 256, 512]), torch.Size([16, 101, 6])
        imgs = imgs.cuda().float()
        imus = imus.cuda().float()
        gts = gts.cuda().float()
        weight = weight.cuda().float()
        timestamps = timestamps.cuda().float()

        optimizer.zero_grad()
        # poses, decisions, probs, _ = model(imgs, imus, is_first=True, hc=None, temp=temp, selection=selection, p=p)
        poses, _ = model(imgs, imus, timestamps, is_first=True, hc=None, temp=temp, selection=selection, p=p)
        if not weighted:
            angle_loss = torch.nn.functional.mse_loss(poses[:,:,:3], gts[:, :, :3])
            translation_loss = torch.nn.functional.mse_loss(poses[:,:,3:], gts[:, :, 3:])
        else:
            weight = weight/weight.sum()
            angle_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:,:,:3] - gts[:, :, :3]) ** 2).mean()
            translation_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:,:,3:] - gts[:, :, 3:]) ** 2).mean()
        
        pose_loss = 100 * angle_loss + translation_loss        
        # penalty = (decisions[:,:,0].float()).sum(-1).mean()
        # loss = pose_loss + args.Lambda * penalty 
        loss = pose_loss
        loss.backward()
        optimizer.step()
        
        if i % args.print_frequency == 0: 
            # message = f'Epoch: {ep}, iters: {i}/{data_len}, pose loss: {pose_loss.item():.6f}, penalty: {penalty.item():.6f}, loss: {loss.item():.6f}'
            message = f'Epoch: {ep}, iters: {i}/{data_len}, pose loss: {pose_loss.item():.6f}, loss: {loss.item():.6f}'
            print(message)
            logger.info(message)

        mse_losses.append(pose_loss.item())
        # penalties.append(penalty.item())

    return np.mean(mse_losses), np.mean(penalties)

def evaluate(model, tester, ep, best):
    # Evaluate the model
    print('Evaluating the model')
    logger.info('Evaluating the model')
    with torch.no_grad(): 
        model.eval()
        errors = tester.eval(model, selection='gumbel-softmax', num_gpu=1)
        tester.generate_plots(checkpoints_dir, ep)
        

    t_rel = np.mean([errors[i]['t_rel'] for i in range(len(errors))])
    r_rel = np.mean([errors[i]['r_rel'] for i in range(len(errors))])
    t_rmse = np.mean([errors[i]['t_rmse'] for i in range(len(errors))])
    r_rmse = np.mean([errors[i]['r_rmse'] for i in range(len(errors))])
    usage = np.mean([errors[i]['usage'] for i in range(len(errors))])

    if t_rel < best:
        best = t_rel 
        torch.save(model.module.state_dict(), f'{checkpoints_dir}/best_{best:.2f}.pth')

    message = f'Epoch {ep} evaluation finished , t_rel: {t_rel:.4f}, r_rel: {r_rel:.4f}, t_rmse: {t_rmse:.4f}, r_rmse: {r_rmse:.4f}, usage: {usage:.4f}, best t_rel: {best:.4f}'
    logger.info(message)
    print(message)
    return best


def main():
    
    # Load the dataset
    transform_train = [ToTensor(),
                       Resize((args.img_h, args.img_w))]
    if args.hflip:
        transform_train += [RandomHorizontalFlip()]
    if args.color:
        transform_train += [RandomColorAug()]
    transform_train = Compose(transform_train)
    signature = inspect.signature(Compose.__call__)
    print(signature) # Check arguments
    
    train_dataset = KITTI(args.data_dir,
                        sequence_length=args.seq_len,
                        train_seqs=args.train_seq,
                        transform=transform_train
                        )
    logger.info('train_dataset: ' + str(train_dataset))
    
    # Using batch sampler here to preserve the sequence boundaries, preventing non-ascending timestamps
    batch_sampler = SequenceBoundarySampler(args.data_dir, batch_size=args.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=args.workers,
        pin_memory=True,
        batch_sampler=batch_sampler
    )
    
    # GPU selections
    gpu_id = set_gpu_ids(args)

    # Model initialization
    model = DeepVIO(args)
    
    # Continual training or not
    load_pretrained_model(model, args)
    
    # Use the pre-trained flownet or not
    if args.pretrain_flownet and args.pretrain is None:
        pretrained_w = torch.load(args.pretrain_flownet, map_location='cpu')
        model_dict = model.Feature_net.state_dict()
        update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
        model_dict.update(update_dict)
        model.Feature_net.load_state_dict(model_dict)
    
    # Initialize the tester
    tester = KITTI_tester(args)

    # Feed model to GPU
    model.cuda(gpu_id)
    model = torch.nn.DataParallel(model, device_ids = [gpu_id])

    pretrain = args.pretrain
    init_epoch = int(pretrain[-7:-4])+1 if args.pretrain is not None else 0    
    
    # Initialize the optimizer
    optimizer = get_optimizer(model, args)
   
    
    best = 10000

    for ep in range(init_epoch, args.epochs_warmup+args.epochs_joint+args.epochs_fine):
        
        lr, selection, temp = update_status(ep, args, model)
        optimizer.param_groups[0]['lr'] = lr
        message = f'Epoch: {ep}, lr: {lr}, selection: {selection}, temperaure: {temp:.5f}'
        print(message)
        logger.info(message)

        model.train()
        avg_pose_loss, avg_penalty_loss = train(model, optimizer, train_loader, selection, temp, logger, ep, p=0.5)
        
        # Save the model after training
        torch.save(model.module.state_dict(), f'{checkpoints_dir}/{ep:003}.pth')
        message = f'Epoch {ep} training finished, pose loss: {avg_pose_loss:.6f}, penalty_loss: {avg_penalty_loss:.6f}, model saved'
        print(message)
        logger.info(message)
        
        # if ep > args.epochs_warmup+args.epochs_joint:
        best = evaluate(model, tester, ep, best)
    
    message = f'Training finished, best t_rel: {best:.4f}'
    logger.info(message)
    print(message)


if __name__ == "__main__":
    main()