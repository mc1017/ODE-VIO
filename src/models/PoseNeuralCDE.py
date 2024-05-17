import torch
import torch.nn as nn
import torchode as to
from src.models.ODEFunc import ODEFunc
from src.models.FusionModule import FusionModule


class PoseNeuralCDE(nn.Module):
    def __init__(self, opt):
        super(PoseNeuralCDE, self).__init__()

        # The main ODE network
        self.f_len = opt.v_f_len + opt.i_f_len
        self.rnn_hidden_size = opt.rnn_hidden_size
        self.fuse_method = opt.fuse_method
        self.rnn_drop_out = opt.rnn_dropout_out
        self.rnn_type = opt.rnn_type

        self.ode_func = ODEFunc(
            feature_dim=self.f_len,
            hidden_dim=opt.ode_hidden_dim,
            num_hidden_layers=opt.ode_num_layers,
            activation=opt.ode_activation_fn,
        )
        self.fuse = FusionModule(feature_dim=self.f_len, fuse_method=self.fuse_method)
        self.rnn = self._set_rnn(opt.rnn_type)
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        # The output regressor network, convert relative hidden state change into relative pose change
        self.regressor = nn.Sequential(
            nn.Linear(self.f_len, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6),
        )

        term = to.ODETerm(self.ode_func)
        step_method = self._set_solver(opt.ode_solver, term=term)
        step_size_controller = to.IntegralController(atol=1e-9, rtol=1e-6, term=term)
        self.solver = torch.compile(
            to.AutoDiffAdjoint(step_method, step_size_controller)
        )

    def _set_solver(self, ode_solver, term):
        solver = None
        if ode_solver == "dopri5":
            solver = to.Dopri5(term=term)
        elif ode_solver == "heun":
            solver = to.Heun(term=term)
        elif ode_solver == "tsit5":
            solver = to.Tsit5(term=term)
        elif ode_solver == "euler":
            solver = to.Euler(term=term)
        else:
            raise ValueError(f"Solver {ode_solver} not supported")
        print("ODE Solver:", ode_solver)
        return solver

    def _set_rnn(self, rnn_type: str):
        rnn = None
        if rnn_type == "rnn":
            rnn = nn.RNNCell(input_size=self.f_len, hidden_size=self.f_len)
        elif rnn_type == "gru":
            rnn = nn.GRUCell(
                input_size=self.f_len, hidden_size=self.f_len
            )
        else:
            raise ValueError(f"RNN type {rnn_type} not supported")
        print("RNN Type:", rnn)
        return rnn

    def forward(self, fv, fv_alter, fi, dec, ts, prev=None, do_profile=False):

        # Fusion of v_in and fi
        v_in = fv
        fused_features = self.fuse(v_in, fi)

        # fused_features.shpae = [batch_size, 1, 768], 768 = 512 (image) + 256 (IMU)
        batch_size, seq_len, _ = fused_features.shape

        initial_hidden_states = (
            prev
            if prev is not None
            else torch.zeros(batch_size, self.f_len).to(fused_features.device)
        )

        if do_profile:
            torch.cuda.nvtx.range_push("odeint")

        problem = to.InitialValueProblem(y0=initial_hidden_states, t_eval=ts)
        ode_solutions = self.solver.solve(problem)

        assert ode_solutions.ys.shape == (batch_size, ts.shape[1], self.f_len)
        new_hidden_states = ode_solutions.ys[:, -1, :].squeeze(1)

        if do_profile:
            torch.cuda.nvtx.range_pop()

        # fused_features.shape = [16, 1, 768], new_hidden_states.shape = [16, 768]

        new_hidden_states = self.rnn(fused_features.squeeze(1), new_hidden_states)
        
        # rnn output shape = [16, 1, 768], rnn_hidden_states shape = [2, batch_size, 768]

        # Dropout layer
        # new_hidden_states = self.rnn_drop_out(new_hidden_states)

        # Since we want to find relative pose changes, we pass in the difference between the new and initial hidden states (new_hidden_states - initial_hidden_states) or stack(new_hidden_states, initial_hidden_states)
        pose = self.regressor(new_hidden_states-initial_hidden_states)

        # pose.shape = [16, 6]
        return pose.unsqueeze(1), new_hidden_states