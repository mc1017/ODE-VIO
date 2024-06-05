import torch
import torch.nn as nn
from src.models.Encoder import ImageEncoder, InertialEncoder
from src.models.PoseODERNN import PoseODERNN
from src.models.PoseRNN import PoseRNN
from src.models.PoseNCP import PoseNCP
from src.models.PoseCDE import PoseCDE
from src.models.PoseRDE import PoseRDE

class DeepVIO(nn.Module):
    """
    Deep Visual-Inertial Odometry (VIO) model that combines data from visual and inertial sensors to estimate poses.

    This class integrates separate encoders for image and inertial data streams, selecting an appropriate model for pose estimation based on configuration options. It supports various pose estimation models such as RNNs, ODE-RNNs, CDEs, and NCPs.

    Attributes:
        Image_net (nn.Module): Image encoder model to process image sequences.
        Inertial_net (nn.Module): Inertial encoder model to process IMU data.
        Pose_net (nn.Module): Dynamically configured pose estimation model based on the specified model type.
        opt (Namespace): Configuration options that include model settings and hyperparameters.

    Methods:
        _set_pose_model(opt): Configures and returns the pose estimation model based on the `model_type` in options.
        forward(img, imu, timestamps, hc=None): Processes the input image and IMU data to compute pose estimations.

    Parameters:
        img (Tensor): Input tensor for image data with dimensions [batch_size, sequence_length, channels, height, width].
        imu (Tensor): Input tensor for inertial measurements with dimensions [batch_size, sequence_length (imu so 10 times img), feature_size].
        timestamps (Tensor): Sequence of timestamps associated with each input sample.
        hc (Tensor, optional): Optional initial hidden state for certain pose models.

    Returns:
        Tuple[Tensor, Tensor]: The estimated poses and the last hidden state from the pose network.
    """
    def __init__(self, opt):
        super(DeepVIO, self).__init__()
        self.Image_net = ImageEncoder(opt)
        self.Inertial_net = InertialEncoder(opt)
        self.Pose_net = self._set_pose_model(opt)
        self.opt = opt
        initialization(self)
    
    def _set_pose_model(self, opt):
        if opt.model_type == "rnn":
            print("Using PoseRNN")
            return PoseRNN(opt)
        elif opt.model_type == "ode-rnn":
            print("Using PoseODERNN")
            return PoseODERNN(opt)
        elif opt.model_type == "cde":
            print("Using PoseCDE")
            return PoseCDE(opt)
        elif opt.model_type == "rde":
            print("Using PoseRDE")
            return PoseRDE(opt)
        elif opt.model_type == "ltc":
            raise NotImplementedError("LTC model not implemented yet")
        
    def forward( self, img, imu, timestamps, hc=None):
        
        # Encode image and imu data
        fv, fi = self.Image_net(img), self.Inertial_net(imu)
        
        # Obtain pose estimations
        poses, h_T = self.Pose_net(fv, fi, timestamps, prev=hc)
        return poses, h_T


def initialization(net):
    for m in net.modules():
        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.Conv1d)
            or isinstance(m, nn.ConvTranspose2d)
            or isinstance(m, nn.Linear)
        ):
            torch.nn.init.kaiming_normal_(m.weight.data)
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
            """
            Note:
            - Xavier uniform initialization is designed to maintain a balanced variance of activations and gradients throughout the network, across different layers during the initial stages of training.
            - Orthogonal initialization ensures that the weight matrices have orthogonal rows (or columns, depending on the dimensionality), which can be beneficial for RNNs including GRUs.
            """
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
