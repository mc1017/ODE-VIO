import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
# from src.utils.plots import plot_flow_and_images


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),  # , inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),  # , inplace=True)
        )


# The inertial encoder for raw imu data
class InertialEncoder(nn.Module):
    def __init__(self, opt):
        super(InertialEncoder, self).__init__()
        self.seq_len = opt.seq_len
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
        )
        self.proj = nn.Linear(256 * 1 * 11, opt.i_f_len)
        self.i_f_len = opt.i_f_len

    def forward(self, x):
        num_pairs = (x.shape[1]-1) // 10
        x = torch.cat(
            [x[:, i * 10 : i * 10 + 11, :].unsqueeze(1) for i in range(num_pairs)],
            dim=1,
        )
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        # x: (N, seq_len, 11, 6)
        x = x.view(
            batch_size * seq_len, x.size(2), x.size(3)
        )  # x: (N x seq_len, 11, 6)
        x = self.encoder_conv(x.permute(0, 2, 1))  # x: (N x seq_len, 64, 11)
        out = self.proj(x.view(x.shape[0], -1))  # out: (N x seq_len, 256)
        return out.view(batch_size, seq_len, self.i_f_len)


class ImageEncoder(nn.Module):
    def __init__(self, opt):
        super(ImageEncoder, self).__init__()
        # CNN
        self.opt = opt
        self.conv1 = conv(True, 6, 64, kernel_size=7, stride=2, dropout=0.2)
        self.conv2 = conv(True, 64, 128, kernel_size=5, stride=2, dropout=0.2)
        self.conv3 = conv(True, 128, 256, kernel_size=5, stride=2, dropout=0.2)
        self.conv3_1 = conv(True, 256, 256, kernel_size=3, stride=1, dropout=0.2)
        self.conv4 = conv(True, 256, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv4_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv5 = conv(True, 512, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv5_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv6 = conv(True, 512, 1024, kernel_size=3, stride=2, dropout=0.5)
        # Comput the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, opt.img_w, opt.img_h))
        __tmp = self.encode_image(__tmp)

        self.visual_head = nn.Linear(int(np.prod(__tmp.size())), opt.v_f_len)

    def forward(self, img):

        # img.shape = [16, 11, 3, 256, 512]
        image_pair = img[:, 0:2, :, :, :]
        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2)
        # v.shape = [16, 10, 6, 256, 512]

        batch_size = v.size(0)
        seq_len = v.size(1)
        
        # image CNN
        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        v = self.encode_image(v)
        v = v.view(batch_size, seq_len, -1)  # (batch, seq_len, fv)
        v = self.visual_head(v)  # (batch, seq_len, 256)
        optical_flow = v[:, :1, :]
        # plot_flow_and_images(image_pair, optical_flow, opt.experiment_name)
        return v

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6
