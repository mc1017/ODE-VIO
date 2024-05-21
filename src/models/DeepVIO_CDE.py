import torch
import torch.nn as nn
import torchcde
from torch.nn.init import kaiming_normal_
from src.models.Encoder import ImageEncoder, InertialEncoder
from src.models.PoseODERNN import PoseODERNN, PoseODERNN_2
from src.models.PoseRNN import PoseRNN
from src.models.PoseNCP import PoseNCP
from src.models.FusionModule import FusionModule
from src.models.ODEFunc import ODEFunc, CDEFunc


class DeepVIO_CDE(nn.Module):
    def __init__(self, opt):
        super(DeepVIO_CDE, self).__init__()
        self.Image_net = ImageEncoder(opt)
        self.Inertial_net = InertialEncoder(opt)
        self.f_len = opt.v_f_len + opt.i_f_len
        self.fuse = FusionModule(self.f_len ,opt.fuse_method)
        self.reduction_net = nn.Linear(self.f_len, opt.cde_hidden_dim)
        self.input_dim = opt.cde_hidden_dim + 1 # reduced feature size + time
        self.cde_hidden_dim = opt.cde_hidden_dim
        self.cde_func = CDEFunc(feature_dim=self.input_dim, 
            hidden_dim=opt.cde_hidden_dim,
            num_hidden_layers=opt.cde_num_layers,
            activation=opt.cde_activation_fn,
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.cde_hidden_dim, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6),
        )
        self.opt = opt
        self.adjoint = opt.adjoint
        initialization(self)
    
    def forward(
        self,
        img,
        imu,
        timestamps,
        is_first=True,
        hc=None,
    ):
        # timestamps [batch_size, 11]
        # Image Size 256x512, specified in args. 3 channels, 11 sequence length, batch size.
        # img.shape = [batch_size, 11, 3, 256, 512] imu.shape[batch_size, 101, 6]
        # Encoder image and imu data. 
        fv, fi = self.Image_net(img), self.Inertial_net(imu)
        # fv.shape = [16, 10, 512] fi.shpae =[16, 10, 256]
        fused_features = self.fuse(fv, fi)
        fused_features = self.reduction_net(fused_features)
        
        
        # Interpolate features to make them continuous
        # Drop the first timestamp of each sequence
        t = timestamps[:, 1:].unsqueeze(-1)  # Now t has shape [batch_size, 10, 1]
        x = torch.cat([t, fused_features], dim=2)  # [batch_size, seq_len, 1 + fused_feature_dim]
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
        X = torchcde.CubicSpline(coeffs)
  
        # Initialize hidden state if first in sequence
        if is_first or hc is None:
            hc = torch.zeros(fv.size(0), self.cde_hidden_dim).to(fused_features.device)

        # Integrate using the Neural CDE
        # print("HC, shape of HC", hc.shape)
        eval_times = torch.arange(0, 10, dtype=torch.float32).to(fused_features.device)
        kwargs = dict(adjoint_params=tuple(self.cde_func.parameters()) + (coeffs, t)) if self.adjoint else {}
        h_T = torchcde.cdeint(X=X, func=self.cde_func, z0=hc, t=eval_times, adjoint=self.adjoint, atol=1e-5, rtol=1e-3, **kwargs)
        poses = self.regressor(h_T)
        h_T = h_T[:, -1, :]  # Take the last hidden state
        return poses, h_T


def initialization(net):
    # Initilization
    for m in net.modules():
        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.Conv1d)
            or isinstance(m, nn.ConvTranspose2d)
            or isinstance(m, nn.Linear)
        ):
            kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif "bias_ih" in name:
                    param.data.fill_(0)
                elif "bias_hh" in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.0)
        elif isinstance(m, nn.GRUCell):
            # Xavier uniform initialization is designed to maintain a balanced variance of activations and gradients throughout the network, across different layers during the initial stages of training.
            # Orthogonal initialization ensures that the weight matrices have orthogonal rows (or columns, depending on the dimensionality), which can be beneficial for RNNs including GRUs.
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)
        elif isinstance(m, nn.RNNCell):
            for name, param in m.named_parameters():
                if "weight_ih" in name or "weight_hh" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
