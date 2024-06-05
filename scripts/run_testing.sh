# ODE-RNN
python3 -m scripts.test_model \
  --experiment_name="74_01drop_vsdataset" \
  --test_name="70%_dropout" \
  --gpu_ids=2 \
  --batch_size=26 \
  --grad_accumulation_steps=1 \
  --ode_activation_fn=softplus \
  --ode_fn_num_layers=2 \
  --rnn_num_layers=3 \
  --ode_hidden_dim=1024 \
  --workers=8 \
  --val_seq 05 07 10 \
  --ode_rnn_type=rnn \
  --fuse_method=soft \
  --freeze_encoder \
  --eval_data_dropout=0.7 \
  --pretrain="/mnt/data0/marco/NeuralCDE-VIO/74_01drop_vsdataset/checkpoints/098.pth" \
  --run_times=10 \