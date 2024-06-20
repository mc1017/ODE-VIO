import torch
import torch.nn as nn
from src.models.FusionModule import FusionModule

    
class PoseRNN(nn.Module):
    """
    Implements an RNN for pose estimation, fusing visual (image) and inertial (IMU) features to estimate pose changes over time.
    
    This model uses an RNN to update the state based on new observations, and a regression network to map the final state to a pose estimate.

    Attributes:
        f_len (int): Total length of fused features combining visual and inertial data.
        rnn_hidden_dim (int): The size of the hidden layer in the RNN.
        fuse_method (str): Method used to fuse visual and inertial features.
        rnn_drop_out (float): Dropout rate for RNN.
        fuse (FusionModule): Module to fuse features.
        rnn (nn.Module): Recurrent neural network module.
        regressor (nn.Sequential): Neural network for regressing from hidden state to pose.

    Methods:
        forward(fv, fi, ts, prev=None, do_profile=False):
            Forward pass for estimating poses from image and IMU features.
            
            Parameters:
                fv (torch.Tensor): Image features [batch_size, seq_len, feature_size].
                fi (torch.Tensor): IMU features [batch_size, seq_len, feature_size].
                ts (torch.Tensor): Timestamps [batch_size, seq_len].
                prev (torch.Tensor, optional): Initial hidden state for the RNN.
                do_profile (bool, optional): Flag to enable profiling.

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Estimated poses and the last hidden state
    """
    def __init__(self, opt):
        super(PoseRNN, self).__init__()

        self.f_len = opt.v_f_len + opt.i_f_len
        self.rnn_hidden_dim = opt.rnn_hidden_dim
        self.rnn_num_layers = opt.rnn_num_layers
        self.fuse_method = opt.fuse_method
        self.rnn_drop_out = opt.rnn_dropout_out

        self.rnn = self._set_rnn(opt.ode_rnn_type)
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        self.fuse = FusionModule(feature_dim=self.f_len, fuse_method=self.fuse_method)
        self.regressor = nn.Sequential(
            nn.Linear(self.f_len, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6),
        )
    
    def forward(self, fv, fi, ts, prev=None, do_profile=False):
        
        # Fuse visual and inertial features
        fused_features = self.fuse(fv, fi)
        batch_size, seq_len, _ = fused_features.shape
        
        # Initialise initial state
        h_0 = torch.zeros(self.rnn_num_layers, batch_size, self.f_len, device=fused_features.device) if prev is None else prev
        
        output = []
        # Evolve the state using the ODE solver
        for i in range(seq_len):
            # Pass the fused features and the evolved hidden states through the RNN
            output_i, rnn_h = self.rnn(fused_features[:, i : i + 1, :], h_0)
            output.append(output_i)
            h_0 = rnn_h
        output = torch.cat(output, dim=1)
        
        # Regress the network to get the pose
        pose = self.regressor(output)
        return pose, h_0


    def _set_rnn(self, rnn_type: str):
        rnn = None
        if rnn_type == "rnn":
            rnn = nn.RNN(input_size=self.f_len, hidden_size=self.f_len, num_layers=self.rnn_num_layers, batch_first=True)
        elif rnn_type == "gru":
            rnn = nn.GRU(input_size=self.f_len, hidden_size=self.f_len, num_layers=self.rnn_num_layers, batch_first=True)
        else:
            raise ValueError(f"RNN type {rnn_type} not supported")
        print("RNN Type:", rnn)
        return rnn
    
    def get_regressor_params(self):
        return self.regressor.parameters()
    
    def get_other_params(self):
        return [param for name, param in self.named_parameters() if not name.startswith('regressor')]