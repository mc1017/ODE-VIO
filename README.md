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

Black formatter and ruff linter are used
```
black .
ruff
```


# Neural CDE

It is worth noting that all of this is very similar to handling irregular data with RNNs, with a few differences:
- Time and observational masks are presented cumulatively, rather than as e.g. delta-time increments.
- It's fine for there to be NaN values in the data (rather than filling them in with zeros or something), because the interpolation routines for torchcde handle that for you.
- Variable length data can be extracted at the end of the CDE, rather than evaluating it at lots of different times. (Incidentally doing so is also more efficient when using the adjoint method, as you only have a single backward solve to make, rather than lots of small ones between all the final times.)

# Activation Functions

SELU activation function is preferred than tanh for intermediate results because of Activation Range and Gradients
1. tanh Activation Function:
Range: The output of the tanh function ranges from -1 to 1. This bounded range can be beneficial for certain types of normalized data, as it helps in keeping the neural network’s output values constrained.
Vanishing Gradients: For large values (either positive or negative), the gradient of the tanh function becomes very small (approaches zero). This phenomenon, known as the vanishing gradient problem, makes it difficult for the model to learn during training, especially for deeper networks, because updates to the weights become insignificantly small.
2. SELU Activation Function:
Self-normalizing Property: SELU has a unique property where it helps in keeping the mean and variance of the outputs of each layer close to zero and one, respectively, during training. This self-normalizing property leads to a more stable and faster convergence.
Non-zero Gradients for Large Inputs: Unlike tanh, SELU maintains non-zero gradients even for large input values, which helps in continuous learning and avoids the vanishing gradient problem common in deep networks.
