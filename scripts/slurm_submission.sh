#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mc620
export PATH=/vol/bitbucket/mc620/NeuralCDE-VIO/venv/bin/:$PATH

source /vol/bitbucket/mc620/NeuralCDE-VIO/venv/bin/activate
source activate
source /vol/cuda/12.2.0/setup.sh
/usr/bin/nvidia-smi
uptime

python -m scripts.train_model --batch_size=26 --experiment_name=58 --ode_activation_fn=softplus --ode_fn_num_layers=3 --ode_hidden_dim=1024 --workers 6 --train_seq 00 02 08 09  --val_seq 00 01 02 04 05 06 07 08 09 10 --rnn_type rnn