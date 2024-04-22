# Advancing VIO with Neural ODEs

This is the repository that contains code for the implmentation of visual-inertial odometry (VIO) algorithm using Neural ODEs. Neural ODEs learns to parameterise a continuous function to fit the trajectory of the device, making it more suitable for handling irregularly sampled data compared to traditional RNNs. 

Since the current SOTA VIO algorithms uses mainly variants of RCNN architectures, imputation or interpolation of data is required in the case of irregular dataset. Neural ODEs provide a promising alternative as the continuous representation naturally handles observations arriving at irregular intervals. 

Here, we provide the implementation of two architectures within the family of Neural ODEs:
1. ODE-RNN
2. Neural CDEs

We use KITTI Odometry dataset as benchmark and compare it against SOTA algorithms. 


## To start training
```
python3 -m scripts.train_model
```

Arguments can be parsed into training using the following format.

```
python3 -m scripts.train_model --gpu_ids=1 --experiment_name=6 --ode_activation_fn=softplus --train_seq 04 10  --val_seq 04 10
```