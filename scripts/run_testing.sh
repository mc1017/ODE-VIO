# ODE-RNN
# python3 -m scripts.test_model \
#   --experiment_name="84_mixed_dropout_03vs_batch_16" \
#   --test_name="50%_dropout" \
#   --gpu_ids=1 \
#   --batch_size=10 \
#   --grad_accumulation_steps=1 \
#   --ode_activation_fn=softplus \
#   --ode_fn_num_layers=2 \
#   --rnn_num_layers=3 \
#   --ode_hidden_dim=1024 \
#   --ode_solver=dopri5 \
#   --workers=2 \
#   --val_seq 05 07 10 \
#   --ode_rnn_type=rnn \
#   --fuse_method=soft \
#   --freeze_encoder \
#   --eval_data_dropout=0.5 \
#   --pretrain="/mnt/data0/marco/NeuralCDE-VIO/84_mixed_dropout_03vs_batch_16/checkpoints/098.pth" \
#   --run_times=5 \


# python3 -m scripts.test_model \
#   --experiment_name="RNN" \
#   --test_name="70%_dropout" \
#   --gpu_ids=2 \
#   --model_type=rnn \
#   --batch_size=10 \
#   --grad_accumulation_steps=1 \
#   --ode_activation_fn=softplus \
#   --ode_fn_num_layers=2 \
#   --rnn_num_layers=3 \
#   --ode_hidden_dim=1024 \
#   --ode_solver=dopri5 \
#   --workers=2 \
#   --val_seq 05 07 10 \
#   --ode_rnn_type=rnn \
#   --fuse_method=soft \
#   --freeze_encoder \
#   --eval_data_dropout=0.7 \
#   --pretrain="/mnt/data0/marco/NeuralCDE-VIO/RNN/checkpoints/086.pth" \
#   --run_times=10 \


python3 -m scripts.test_model \
  --experiment_name="158-not-return-states-save-observations" \
  --test_name="0%_dropout" \
  --gpu_ids=2 \
  --model_type=cde \
  --batch_size=10 \
  --model_type=cde \
  --grad_accumulation_steps=1 \
  --cde_activation_fn=softplus \
  --cde_solver=dopri5 \
  --cde_fn_num_layers=2 \
  --cde_num_layers=1 \
  --v_f_len=128 \
  --i_f_len=64 \
  --cde_hidden_dim=192 \
  --workers=2 \
  --val_seq 04 \
  --fuse_method=cat \
  --freeze_encoder \
  --eval_data_dropout=0.0 \
  --pretrain="/mnt/data0/marco/NeuralCDE-VIO/158-not-return-states-save-observations/checkpoints/026.pth" \
  --run_times=10 \