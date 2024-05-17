import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from src.models.Encoder import ImageEncoder, InertialEncoder
from src.models.PoseODERNN import PoseODERNN



class DeepVIO(nn.Module):
    def __init__(self, opt):
        super(DeepVIO, self).__init__()
        self.Image_net = ImageEncoder(opt)
        self.Inertial_net = InertialEncoder(opt)
        self.Pose_net = PoseODERNN(opt)
        self.opt = opt
        initialization(self)

    def forward(
        self,
        img,
        imu,
        timestamps,
        is_first=True,
        hc=None,
    ):
        # Image Size 256x512, specified in args. 3 channels, 11 sequence length, batch size 16
        # img.shape = [16, 11, 3, 256, 512] imu.shape[16, 101, 6]
        fv, fi = self.Image_net(img), self.Inertial_net(imu)
        # fv.shape = [16, 10, 512] fi.shpae =[16, 10, 256]
        seq_len = fv.shape[1]

        poses = []
        for i in range(seq_len):
            # fv.shape = [16, 10, 512], fi.shape = [16, 10, 256], timestamps.shape = [16, 11]
            pose, hc = self.Pose_net(
                fv[:, i : i + 1, :],
                None,
                fi[:, i : i + 1, :],
                None,
                timestamps[:, i : i + 2],
                prev=hc,
            )
            poses.append(pose)
        poses = torch.cat(poses, dim=1)
        return poses, hc.detach()


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
