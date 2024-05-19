#!/bin/bash

# Run the training script with specified arguments
python -m scripts.train_model \
  --experiment_name=59 \
  --gpu_ids=0 \
  --batch_size=26 \
  --grad_accumulation_steps=1 \
  --ode_activation_fn=softplus \
  --ode_num_layers=3 \
  --ode_hidden_dim=1024 \
  --workers=6 \
  --train_seq 00 02 08 09 \
  --val_seq 01 04 05 06 07 10 \
  --rnn_type=rnn \
  --fuse_method=soft \
  --freeze_encoder \
  --wandb \
