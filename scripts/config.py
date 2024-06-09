# config.py

import argparse

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths for data and results
    parser.add_argument("--data_dir", type=str, default="/mnt/data0/marco/KITTI/data", help="path to the dataset")
    parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
    parser.add_argument("--save_dir", type=str, default="./results", help="path to save the result")
    parser.add_argument("--plot_dir", type=str, default="./results", help="path to save the log")

    # Training Configurations
    parser.add_argument("--experiment_name", type=str, default="experiment", help="experiment name")
    parser.add_argument("--wandb", default=False, action="store_true", help="whether to use wandb logging")
    parser.add_argument("--wandb_group", type=str, default="ode-rnn", help="group of the wandb run")
    parser.add_argument("--sweep", default=False, action="store_true", help="whether to use wandb sweep")
    parser.add_argument("--resume", type=str, default=None, help="resume training (wandb run id)")
    parser.add_argument("--pretrain_flownet", type=str, default="./pretrained_models/flownets_bn_EPE2.459.pth.tar", help="whether to use the pre-trained flownet")
    parser.add_argument("--pretrain", type=str, default=None, help="path to the pretrained model")
    parser.add_argument("--train_seq", type=str, default=["00", "01", "02", "04", "08", "09"], nargs="+", help="sequences for training")
    parser.add_argument("--val_seq", type=str, default=["06"], nargs="+", help="sequences for validation")
    parser.add_argument("--test_seq", type=str, default=["05", "07", "10"], nargs="+", help="sequences for testing")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--workers", type=int, default=8, help="number of workers in dataloader")
    parser.add_argument("--print_frequency", type=int, default=10, help="print frequency for loss values")

    # Training Hyperparameters
    parser.add_argument("--model_type", type=str, default="ode-rnn", help="type of model [rnn, ode, ode-rnn, ltc, cfc]")
    parser.add_argument("--optimizer", type=str, default="Adam", help="type of optimizer [Adam, SGD]")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="gradient accumulation steps before updating")
    parser.add_argument("--freeze_encoder", default=False, action="store_true", help="freeze the encoder or not")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="weight decay for the optimizer")
    parser.add_argument("--batch_size", type=int, default=26, help="batch size")
    parser.add_argument("--shuffle", type=int, default=True, help="shuffle data samples or not")
    parser.add_argument("--epochs_warmup", type=int, default=20, help="number of epochs for warmup")
    parser.add_argument("--epochs_joint", type=int, default=40, help="number of epochs for joint training")
    parser.add_argument("--epochs_fine", type=int, default=40, help="number of epochs for finetuning")
    parser.add_argument("--lr_warmup", type=float, default=1e-4, help="learning rate for warming up stage")
    parser.add_argument("--lr_joint", type=float, default=1e-5, help="learning rate for joint training stage")
    parser.add_argument("--lr_fine", type=float, default=1e-6, help="learning rate for finetuning stage")
    parser.add_argument("--gradient_clip", type=float, default=5, help="gradient clipping norm/clip value")

    # Data Augmentation
    parser.add_argument("--data_dropout", type=float, default=0.0, help="irregularity in the dataset by dropping out randomly")
    parser.add_argument("--data_dropout_std", type=float, default=0.0, help="std of irregularity across each epoch")
    parser.add_argument("--eval_data_dropout", type=float, default=0.0, help="irregularity in the eval dataset")
    parser.add_argument("--img_w", type=int, default=512, help="image width")
    parser.add_argument("--img_h", type=int, default=256, help="image height")
    parser.add_argument("--v_f_len", type=int, default=512, help="visual feature length")
    parser.add_argument("--i_f_len", type=int, default=256, help="imu feature length")
    parser.add_argument("--imu_dropout", type=float, default=0, help="dropout for the IMU encoder")
    parser.add_argument("--hflip", default=False, action="store_true", help="whether to use horizontal flipping as augmentation")
    parser.add_argument("--color", default=False, action="store_true", help="whether to use color augmentations")
    parser.add_argument("--seq_len", type=int, default=11, help="sequence length of images")
    parser.add_argument("--normalize", default=False, action="store_true", help="whether to normalize the images")

    # Fusion Module Parameters
    parser.add_argument("--fuse_method", type=str, default="cat", help="fusion method of encoded IMU and Images [cat, soft, hard]")

    # ODE Parameters
    parser.add_argument("--ode_hidden_dim", type=int, default=512, help="size of the ODE latent")
    parser.add_argument("--ode_fn_num_layers", type=int, default=3, help="number of layers for the ODE")
    parser.add_argument("--ode_activation_fn", type=str, default="tanh", help="activation function [softplus, relu, leaky_relu, tanh]")
    parser.add_argument("--ode_solver", type=str, default="dopri5", help="ODE solvers [dopri5, heun, euler, runge_kutta, tsit5]")

    # RNN Regressor Parameters for ODE-RNN/NCP Implementation
    parser.add_argument("--ode_rnn_type", type=str, default="rnn", help="type of RNN [rnn, lstm, gru]")
    parser.add_argument("--rnn_num_layers", type=int, default=2, help="number of layers for RNN")
    parser.add_argument("--rnn_hidden_dim", type=int, default=1024, help="size of the RNN latent")
    parser.add_argument("--rnn_dropout_out", type=float, default=0, help="dropout for the RNN output layer")

    # CDE Parameters
    parser.add_argument("--cde_hidden_dim", type=int, default=128, help="size of the CDE latent")
    parser.add_argument("--cde_fn_num_layers", type=int, default=3, help="number of layers for the CDE Function")
    parser.add_argument("--cde_num_layers", type=int, default=3, help="number of layers for the CDE")
    parser.add_argument("--cde_activation_fn", type=str, default="tanh", help="activation function [softplus, relu, leaky_relu, tanh]")
    parser.add_argument("--cde_solver", type=str, default="dopri5", help="ODE solvers [dopri5, heun, euler, runge_kutta, tsit5]")
    parser.add_argument("--adjoint", default=False, action="store_true", help="whether to use adjoint method")

    args = parser.parse_args()
    return args
