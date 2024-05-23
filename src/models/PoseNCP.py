import torch
import torch.nn as nn
from src.models.FusionModule import FusionModule
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP, FullyConnected

class PoseNCP(nn.Module):
    def __init__(self, opt):
        super(PoseNCP, self).__init__()

        # The main network configuration
        self.f_len = opt.v_f_len + opt.i_f_len
        self.rnn_hidden_dim = opt.rnn_hidden_dim
        self.fuse_method = opt.fuse_method
        self.rnn_drop_out = opt.rnn_dropout_out
        self.ncp_type = opt.model_type

        self.fuse = FusionModule(feature_dim=self.f_len, fuse_method=self.fuse_method)

        # Configure the NCP wiring
        wiring = FullyConnected(self.f_len, self.rnn_hidden_dim)

        # Choose NCP RNN type
        if self.ncp_type == 'ltc':
            self.rnn = LTC(self.f_len, wiring, batch_first=True)
        elif self.rnn_type == 'cfc':
            self.rnn = CfC(self.f_len, wiring, batch_first=True)
        else:
            raise ValueError(f"RNN type {self.ncp_type} not supported")

        # The output regressor network, convert relative hidden state change into relative pose change
        self.regressor = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6),
        )

    def forward(self, fv, fv_alter, fi, dec, ts, prev=None, do_profile=False):

        # Fusion of v_in and fi
        v_in = fv
        fused_features = self.fuse(v_in, fi)

        # fused_features.shpae = [batch_size, 1, 768], 768 = 512 (image) + 256 (IMU)
        batch_size, seq_len, _ = fused_features.shape

        initial_hidden_states = (
            prev
            if prev is not None
            else torch.zeros(batch_size, self.rnn_hidden_dim).to(fused_features.device)
        )

        if do_profile:
            torch.cuda.nvtx.range_push("ncp_forward")

        # Pass through the NCP RNN
        relative_ts = ts[:, 1:] - ts[:, :-1]
        new_hidden_states, _ = self.rnn(fused_features, initial_hidden_states, timespans=relative_ts)
        print("New Hidden States Shape:", new_hidden_states.shape)

        if do_profile:
            torch.cuda.nvtx.range_pop()

        # new_hidden_states = self.rnn_drop_out(new_hidden_states)

        # Since we want to find relative pose changes, we pass in the difference between the new and initial hidden states (new_hidden_states - initial_hidden_states) or stack(new_hidden_states, initial_hidden_states)
        pose = self.regressor(new_hidden_states - initial_hidden_states)

        return pose.unsqueeze(1), new_hidden_states
