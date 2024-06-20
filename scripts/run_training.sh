#!/bin/bash

# Run the training script with specified arguments

# ODE-RNN
python3 -m scripts.train_model \
  --experiment_name=testing \
  --gpu_ids=1 \
  --batch_size=16 \
  --grad_accumulation_steps=1 \
  --ode_activation_fn=softplus \
  --ode_fn_num_layers=2 \
  --ode_solver=dopri5 \
  --rnn_num_layers=3 \
  --ode_hidden_dim=1024 \
  --workers=2 \
  --train_seq 00 01 02 04 06 08 09 \
  --val_seq 05 07 10 \
  --lr_warmup=1e-4 \
  --lr_joint=1e-5 \
  --lr_fine=1e-6 \
  --ode_rnn_type=rnn \
  --fuse_method=soft \
  --freeze_encoder \
  --data_dropout=0.2 \
  --data_dropout_std=0.1 \
  --eval_data_dropout=0.2 \
  --print_frequency=5 \
#   --wandb \
#   --wandb_group="ode-rnn" \
#   --pretrain="/mnt/data0/marco/NeuralCDE-VIO/85_mixed_dropout_02vs_batch_16/checkpoints/090.pth" \


# Neural CDE
# python3 -m scripts.train_model \
#   --experiment_name=test-memory \
#   --gpu_ids=1 \
#   --batch_size=1 \
#   --model_type=cde \
#   --grad_accumulation_steps=1 \
#   --cde_activation_fn=softplus \
#   --cde_solver=dopri5 \
#   --cde_fn_num_layers=2 \
#   --cde_num_layers=1 \
#   --v_f_len=200 \
#   --i_f_len=200 \
#   --cde_hidden_dim=400 \
#   --workers=2 \
#   --train_seq 10 \
#   --val_seq 10 \
#   --shuffle=True \
#   --lr_warmup=5e-5 \
#   --lr_joint=5e-6 \
#   --lr_fine=5e-7 \
#   --epochs_warmup=20 \
#   --epochs_joint=40 \
#   --epochs_fine=40 \
#   --fuse_method=cat \
#   --freeze_encoder \
#   --wandb_group="cde" \
#   --print_frequency=10 \
#   --seq_len=11 \
#   --print_frequency=5 \
#   --adjoint \
  # --wandb \
#   --pretrain="/mnt/data0/marco/NeuralCDE-VIO/152-hidden-return-first/checkpoints/020.pth" \

# RDE
# python3 -m scripts.train_model \
#   --experiment_name=139 \
#   --gpu_ids=1 \
#   --batch_size=10 \
#   --model_type=cde \
#   --grad_accumulation_steps=1 \
#   --cde_activation_fn=relu \
#   --cde_solver=dopri5 \
#   --cde_fn_num_layers=2 \
#   --cde_num_layers=1 \
#   --cde_hidden_dim=768 \
#   --workers=8 \
#   --train_seq 00 01 02 04 06 08 09 \
#   --val_seq 05 07 10 \
#   --lr_warmup=5e-5 \
#   --lr_joint=5e-6 \
#   --lr_fine=5e-7 \
#   --fuse_method=cat \
#   --freeze_encoder \
#   --wandb_group="rde" \
#   --gradient_clip=1 \
#   --print_frequency=5 \
#   --seq_len=21 \
  # --adjoint \


# RNN
# python3 -m scripts.train_model \
#   --experiment_name=RNN \
#   --gpu_ids=1 \
#   --batch_size=32 \
#   --model_type=rnn \
#   --grad_accumulation_steps=1 \
#   --rnn_num_layers=3 \
#   --workers=2 \
#   --train_seq 00 01 02 04 06 08 09 \
#   --val_seq 05 07 10 \
#   --lr_warmup=1e-4 \
#   --lr_joint=1e-5 \
#   --lr_fine=1e-6 \
#   --fuse_method=soft \
#   --freeze_encoder \
#   --data_dropout=0.3 \
#   --data_dropout_std=0.1 \
#   --eval_data_dropout=0.3 \
#   --seq_len=11 \
#   --wandb \
#   --wandb_group="cde" \
