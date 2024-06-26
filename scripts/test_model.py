import argparse
import torch
import numpy as np
from src.data.KITTI_eval import KITTI_tester
from src.models.DeepVIO import DeepVIO
from utils.utils import set_gpu_ids 
from pathlib import Path
import os


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument( "--data_dir", type=str, default="/mnt/data0/marco/KITTI/data", help="path to the dataset",)
parser.add_argument( "--checkpoint_dir", type=str, default="/mnt/data0/marco/NeuralCDE-VIO", help="path to the checkpoints",)
parser.add_argument( "--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU",)
parser.add_argument( "--save_dir", type=str, default="./results", help="path to save the result")
parser.add_argument( "--plot_dir", type=str, default="./results", help="path to save the log")
parser.add_argument( "--run_times", type=int, default=1, help="number of times to run the experiment")

# Training Configurations
parser.add_argument( "--experiment_name", type=str, default="experiment", help="experiment name")
parser.add_argument( "--test_name", type=str, default="0%_dropout", help="experiment name")
parser.add_argument( "--pretrain_flownet", type=str, default="./pretrained_models/flownets_bn_EPE2.459.pth.tar", help="wehther to use the pre-trained flownet",)
parser.add_argument( "--pretrain", type=str, default=None, help="path to the pretrained model")
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
parser.add_argument( "--shuffle", type=int, default=True, help="shuffle data samples or not")
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
parser.add_argument( "--ode_fn_num_layers", type=int, default=3, help="number of layers for the ODE function")
parser.add_argument( "--ode_activation_fn", type=str, default="tanh", help="activation function [softplus, relu, leaky_relu, tanh]",)
parser.add_argument( "--ode_solver", type=str, default="dopri5", help="ODE solvers [dopri5, heun, euler, runge_kutta, tsit5]",)

# RNN Regressor Parameters for ODE-RNN/NCP Implementation
parser.add_argument( "--ode_rnn_type", type=str, default="rnn", help="type of RNN [rnn, lstm, gru]") 
parser.add_argument( "--rnn_num_layers", type=int, default=2, help="number of layers for RNN")
parser.add_argument( "--rnn_hidden_dim", type=int, default=1024, help="size of the RNN latent") 
parser.add_argument( "--rnn_dropout_out", type=float, default=0, help="dropout for the RNN output layer",)

# CDE Parameters 
parser.add_argument( "--cde_hidden_dim", type=int, default=128, help="size of the CDE latent")
parser.add_argument( "--cde_fn_num_layers", type=int, default=3, help="number of layers for the CDE Function")
parser.add_argument( "--cde_num_layers", type=int, default=3, help="number of layers for the cDE")
parser.add_argument( "--cde_activation_fn", type=str, default="tanh", help="activation function [softplus, relu, leaky_relu, tanh]",)
parser.add_argument( "--cde_solver", type=str, default="dopri5", help="ODE solvers [dopri5, heun, euler, runge_kutta, tsit5]",)
parser.add_argument( "--adjoint", default=False, action="store_true", help="whether to use adjoint method",)


args = parser.parse_args()

# Set the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def setup_test_directory():
    experiment_dir = Path("./results")
    experiment_dir.mkdir(exist_ok=True)
    file_dir = experiment_dir.joinpath(args.experiment_name)
    file_dir.mkdir(exist_ok=True)
    test_dir = file_dir.joinpath("test", args.test_name)
    test_dir.mkdir(exist_ok=True, parents=True)
    return test_dir
    
    

def main():
    test_dir = setup_test_directory()
    
    # GPU selections
    gpu_id = set_gpu_ids(args)
    
    # Initialize a dictionary to store errors for each sequence
    results = {seq: {'t_rel': [], 'r_rel': [], 't_rmse': [], 'r_rmse': [], 'usage': []} for seq in args.val_seq} 
    print(args.run_times)
    
    for j in range(args.run_times):
        # Initialize the tester
        tester = KITTI_tester(args)

        # Model initialization
        model = DeepVIO(args)

        model.load_state_dict(torch.load(args.pretrain))
        print('load model %s'%args.pretrain)
        print('Dropout percentage:', args.eval_data_dropout)
            
        # Feed model to GPU
        model.cuda(gpu_id)
        model.eval()

        errors = tester.eval(model, num_gpu=1)
        tester.generate_plots(test_dir, 30)
        tester.save_text(test_dir)
        
        for i, seq in enumerate(args.val_seq):
            # Collect errors for each sequence
            results[seq]['t_rel'].append(tester.errors[i]['t_rel'])
            results[seq]['r_rel'].append(tester.errors[i]['r_rel'])
            results[seq]['t_rmse'].append(tester.errors[i]['t_rmse'])
            results[seq]['r_rmse'].append(tester.errors[i]['r_rmse'])
            message = f"Seq: {seq}, t_rel: {tester.errors[i]['t_rel']:.4f}, r_rel: {tester.errors[i]['r_rel']:.4f}, "
            message += f"t_rmse: {tester.errors[i]['t_rmse']:.4f}, r_rmse: {tester.errors[i]['r_rmse']:.4f}, "
            print(message)
    return results
            

if __name__ == "__main__":
    results = main()
    test_dir = setup_test_directory()
    output_file_path = os.path.join(test_dir, "summary.txt")
    with open(output_file_path, 'w') as f:
        for seq in args.val_seq:
            t_rel_mean = np.mean(results[seq]['t_rel'])
            t_rel_std = np.std(results[seq]['t_rel'])
            r_rel_mean = np.mean(results[seq]['r_rel'])
            r_rel_std = np.std(results[seq]['r_rel'])
            t_rmse_mean = np.mean(results[seq]['t_rmse'])
            t_rmse_std = np.std(results[seq]['t_rmse'])
            r_rmse_mean = np.mean(results[seq]['r_rmse'])
            r_rmse_std = np.std(results[seq]['r_rmse'])
            
            summary_message = (f"Seq: {seq}, t_rel_mean: {t_rel_mean:.4f} (std: {t_rel_std:.4f}), "
                            f"r_rel_mean: {r_rel_mean:.4f} (std: {r_rel_std:.4f}), "
                            f"t_rmse_mean: {t_rmse_mean:.4f} (std: {t_rmse_std:.4f}), "
                            f"r_rmse_mean: {r_rmse_mean:.4f} (std: {r_rmse_std:.4f}), ")
            
            print(summary_message)
            f.write(summary_message + '\n')
        
