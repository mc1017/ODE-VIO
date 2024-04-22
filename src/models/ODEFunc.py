import torch.nn as nn


# ODE Function for Neural ODE
class ODEFunc(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_hidden_layers=3, actiavtion="tanh"):
        super(ODEFunc, self).__init__()
        activation_func = self._set_activation(actiavtion)
        layers = [nn.Linear(feature_dim, hidden_dim), activation_func]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_func)
        layers.append(nn.Linear(hidden_dim, feature_dim))
        layers.append(nn.Tanh())  # Tanh activation layer at last improves training
        self.net = nn.Sequential(*layers)

        # Initialize weights and biases
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def _set_activation(self, ode_activation_fn):
        activation = None
        if ode_activation_fn == "tanh":
            activation = nn.Tanh()
        elif ode_activation_fn == "relu":
            activation = nn.ReLU()
        elif ode_activation_fn == "leaky_relu":
            activation == nn.LeakyReLU()
        elif ode_activation_fn == "softplus":
            activation == nn.Softplus()
        else:
            raise ValueError(f"Activation function {ode_activation_fn} not supported")
        print("ODE Activation Function:", ode_activation_fn)
        return activation
    
    def forward(self, t, x):
        return self.net(x)
