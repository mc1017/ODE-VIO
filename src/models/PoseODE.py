import torch
import torch.nn as nn
from src.models.ODEFunc import ODEFunc
from src.models.FusionModule import FusionModule
from torchdiffeq import odeint


# The pose estimation network
class PoseODE(nn.Module):
    def __init__(self, opt):
        super(PoseODE, self).__init__()

        # The main ODE network
        self.f_len = opt.v_f_len + opt.i_f_len
        self.ode_func = ODEFunc(hidden_dim=self.f_len)
        self.fuse = FusionModule(opt)

        # The output networks
        self.regressor = nn.Sequential(
            nn.Linear(self.f_len, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6),
        )

    def forward(self, fv, fv_alter, fi, dec, ts, prev=None):

        # Fusion of v_in and fi
        v_in = fv
        fused_features = self.fuse(v_in, fi)

        # Initial state for ODE solver, potentially replacing zeros with a more meaningful initial state
        batch_size, seq_len, _ = fused_features.shape
        # if prev is not None:
        #     print('Prev:, ', prev.shape, prev)
        initial_state = (
            prev
            if prev is not None
            else torch.zeros(batch_size, self.f_len).to(fused_features.device)
        )
        integrated_states = initial_state

        for i in range(seq_len):
            # Incorporated time, but using dopri5 result in NaNs. Use fixed step size
            # fixed_adams https://github.com/rtqichen/torchdiffeq/issues/27
            integrated_states = odeint(
                self.ode_func, integrated_states, ts[i : i + 2], method="fixed_adams"
            )[-1]
        # print(ts.shape)
        # print(integrated_states.shape)

        pose = self.regressor(integrated_states).unsqueeze(1)
        # print("Poses: ", pose.shape)
        return pose, integrated_states
