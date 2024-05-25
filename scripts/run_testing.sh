# ODE-RNN
python3 -m scripts.test_model \
  --experiment_name="72 - 0 Drop, separate test" \
  --test_name="10%_dropout" \
  --gpu_ids=0 \
  --batch_size=26 \
  --grad_accumulation_steps=1 \
  --ode_activation_fn=softplus \
  --ode_num_layers=2 \
  --rnn_num_layers=3 \
  --ode_hidden_dim=1024 \
  --workers=8 \
  --val_seq 05 07 10 \
  --ode_rnn_type=rnn \
  --fuse_method=soft \
  --freeze_encoder \
  --eval_data_dropout=0.2 \
  --pretrain="/vol/bitbucket/mc620/NeuralCDE-VIO/72 - 0 Drop, separate test/checkpoints/058.pth" \