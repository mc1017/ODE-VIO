import torch.nn as nn
from src.models import FusionModule


# The pose estimation network
class Pose_RNN(nn.Module):
    def __init__(self, opt):
        super(Pose_RNN, self).__init__()

        # The main RNN network
        f_len = opt.v_f_len + opt.i_f_len
        self.rnn = nn.LSTM(
            input_size=f_len,
            hidden_size=opt.rnn_hidden_size,
            num_layers=2,
            dropout=opt.rnn_dropout_between,
            batch_first=True,
        )

        self.fuse = FusionModule(opt)

        # The output networks
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        self.regressor = nn.Sequential(
            nn.Linear(opt.rnn_hidden_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6),
        )

    def forward(self, fv, fv_alter, fi, dec, prev=None):
        if prev is not None:
            prev = (
                prev[0].transpose(1, 0).contiguous(),
                prev[1].transpose(1, 0).contiguous(),
            )

        # Select between fv and fv_alter
        # The first element of decision determines whether to use fv and last element determine to use fv_alter
        v_in = (
            fv * dec[:, :, :1] + fv_alter * dec[:, :, -1:]
            if fv_alter is not None
            else fv
        )
        fused = self.fuse(v_in, fi)

        out, hc = self.rnn(fused) if prev is None else self.rnn(fused, prev)
        out = self.rnn_drop_out(out)
        pose = self.regressor(out)

        hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
        # pose.shape = [16, 1, 6]
        # hc[0].shape, hc[1].shape = (16, 2, 1024), (16, 2, 1024)
        return pose, hc
