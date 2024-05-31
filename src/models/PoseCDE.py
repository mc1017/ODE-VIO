import torch
import torch.nn as nn
import torchcde as cde
from src.models.ODEFunc import ODEFunc, CDEFunc
from src.models.FusionModule import FusionModule

class PoseCDE(nn.Module):
    """
    A module for pose estimation using Neural Controlled Differential Equations (CDEs).

    This module combines visual (image) and inertial (IMU) features, integrates them over time using a neural CDE approach, and regresses to pose estimations. It employs spline interpolation to make feature sequences continuous before applying neural CDEs to integrate hidden states.

    Attributes:
        f_len (int): Total length of fused features.
        input_dim (int): Input dimension to the CDE function, calculated as reduced feature size plus time.
        cde_hidden_dim (int): Hidden dimension size for the CDE function.
        cde_func (CDEFunc): The neural CDE function configured with specific feature and hidden dimensions.
        fuse (FusionModule): Module to fuse image and IMU features.
        reduction_net (nn.Linear): Linear layer to reduce dimensionality of fused features.
        initial_net (nn.Linear): If initial state is not provided, it should depend on the initial observation
        regressor (nn.Sequential): Sequence of layers to regress the final hidden state to pose estimates.
        opt (Namespace): Configuration options passed as an argument.
        adjoint (bool): Flag to use adjoint method for backpropagation to reduce memory usage.
    
    Methods:
        forward(fv, fi, ts, prev=None, do_profile=False):
            Processes batches of image and IMU features along with timestamps to estimate poses.

            Parameters:
                fv (torch.Tensor): Image features of shape [batch_size, seq_len-1, feature_size].
                fi (torch.Tensor): IMU features of shape [batch_size, seq_len-1, feature_size].
                ts (torch.Tensor): Timestamps of shape [batch_size, seq_len].
                prev (torch.Tensor, optional): Previous hidden state for the neural CDE. Defaults to None.
                do_profile (bool, optional): Flag to enable profiling of the forward pass. Defaults to False.
                Note: fv and fi have seq_len-1 is because we are taking pairwise difference between obserations

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Estimated poses and the last hidden state from the CDE integration.
    """
    
    def __init__(self, opt):
        super(PoseCDE, self).__init__()
        
        self.opt = opt
        self.adjoint = opt.adjoint
        self.f_len = opt.v_f_len + opt.i_f_len
        self.input_dim = opt.cde_hidden_dim + 1 # reduced feature size + time
        self.cde_hidden_dim = opt.cde_hidden_dim
        self.cde_num_layers = opt.cde_num_layers
        self.cde_fn_num_layers = opt.cde_fn_num_layers
      
        self.fuse = FusionModule(self.f_len ,opt.fuse_method)
        self.reduction_net = nn.Linear(self.f_len, opt.cde_hidden_dim)
        self.initial = nn.Linear(opt.cde_hidden_dim+1, opt.cde_hidden_dim)
        self.cde_func = CDEFunc(feature_dim=self.input_dim, 
            hidden_dim=opt.cde_hidden_dim,
            num_hidden_layers=opt.cde_fn_num_layers,
            activation=opt.cde_activation_fn,
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.cde_hidden_dim, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6),

        )
        self.solver = opt.cde_solver
       
    
    def forward(self, fv, fi, ts, prev=None, do_profile=False):
        # Fuse and reduce feature size
        fused_features = self.fuse(fv, fi)
        # fused_features = self.reduction_net(fused_features)
        batch_size, seq_len, _ = fused_features.shape
        # print("max, min, mean, median, std:", fused_features.max().item(), fused_features.min().item(), fused_features.mean().item(), fused_features.median().item(), fused_features.std().item())
            
        # Subtract the first timestamp from all timestamps to get time differences
        # Remove the first timestamp and add a dimension
        ts_diff = ts - ts[:, :1] 
        ts_diff = ts_diff[:, 1:].unsqueeze(-1) 
        
        h_T = []
        h_0 = torch.zeros(self.cde_num_layers, batch_size, self.cde_hidden_dim, device=fused_features.device) if prev is None else prev
        for i in range(self.cde_num_layers): 
            x = torch.cat([ts_diff, fused_features], dim=-1)
            coeffs = cde.linear_interpolation_coeffs(x, rectilinear=0)
            X = cde.LinearInterpolation(coeffs)
            # X0 = X.evaluate(X.interval[0])
            
            # Create a tensor from 0.01 to 1.0 with a step of 0.01
            eval_times = torch.linspace(0.1, 1.0, 10, dtype=torch.float32).to(fused_features.device)

            # Integrate using the Neural CDE with all evaluation timestamps
            kwargs = dict(adjoint_params=tuple(self.cde_func.parameters()) + (coeffs, ts_diff)) if self.adjoint else {}
            h_i = cde.cdeint(X=X, func=self.cde_func, z0=h_0[i], t=eval_times, adjoint=self.adjoint, atol=1e-6, rtol=1e-4, method=self.solver, **kwargs)
            fused_features = h_i
            h_T.append(h_i[:, -1, :])

        # Regress the relative poses using the extracted steps
        # print(h_i.shape,h_0.shape,  torch.diff(h_i, dim=1).shape)?
        h_diff = torch.cat([(h_i[:,0,:]-h_0[0]).unsqueeze(1), torch.diff(h_i, dim=1)], dim=1)
        
        # print("h_diff shape:", h_diff.shape)
        poses = self.regressor(h_diff)
        h_T = torch.stack(h_T, dim=0)
        return poses, h_T # Return the last hidden state
    
    def get_regressor_params(self):
        return self.regressor.parameters()
    
    def get_other_params(self):
        return [param for name, param in self.named_parameters() if not name.startswith('regressor')]