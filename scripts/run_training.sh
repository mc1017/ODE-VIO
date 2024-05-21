#!/bin/bash

# Run the training script with specified arguments

# ODE-RNN
python3 -m scripts.train_model \
  --experiment_name="63 - 3 rnn layer, output regress" \
  --gpu_ids=1 \
  --batch_size=26 \
  --grad_accumulation_steps=1 \
  --ode_activation_fn=softplus \
  --ode_num_layers=4 \
  --rnn_num_layers=3 \
  --ode_hidden_dim=1024 \
  --workers=8 \
  --train_seq 00 02 08 09 \
  --val_seq 01 04 05 06 07 10 \
  --ode_rnn_type=rnn \
  --fuse_method=soft \
  --freeze_encoder \
  --wandb \

  # --pretrain="/mnt/data0/marco/NeuralCDE-VIO/61/checkpoints/016.pth" \
  # --resume="zaq8x9r1" \

# NCP (LTC/CfC)
# python -m scripts.train_model \
#   --experiment_name=100 \
#   --gpu_ids=2 \
#   --batch_size=26 \
#   --grad_accumulation_steps=1 \
#   --workers=6 \
#   --model_type=ltc \
#   --train_seq 00 02 08 09 \
#   --val_seq 01 04 05 06 07 10 \
#   --ode_rnn_type=rnn \
#   --fuse_method=hard \
#   --freeze_encoder \
  # --wandb \