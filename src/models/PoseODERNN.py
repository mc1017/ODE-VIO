import torch
import torch.nn as nn
import torchode as to
from src.models.ODEFunc import ODEFunc
from src.models.FusionModule import FusionModule

    
class PoseODERNN(nn.Module):
    """
    Implements an RNN combined with an Ordinary Differential Equation (ODE) solver for pose estimation, fusing visual (image) and inertial (IMU) features to estimate pose changes over time.
    
    This model uses a configurable ODE function to evolve the state of the system, an RNN to update the state based on new observations, and a regression network to map the final state to a pose estimate.

    Attributes:
        f_len (int): Total length of fused features combining visual and inertial data.
        rnn_hidden_dim (int): The size of the hidden layer in the RNN.
        fuse_method (str): Method used to fuse visual and inertial features.
        rnn_drop_out (float): Dropout rate for RNN.
        ode_func (ODEFunc): Configured ODE function for state evolution.
        fuse (FusionModule): Module to fuse features.
        rnn (nn.Module): Recurrent neural network module.
        solver (torchode.Solver): ODE solver for continuous integration of state.
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
        super(PoseODERNN, self).__init__()

        self.f_len = opt.v_f_len + opt.i_f_len
        self.rnn_hidden_dim = opt.rnn_hidden_dim
        self.rnn_num_layers = opt.rnn_num_layers
        self.fuse_method = opt.fuse_method
        self.rnn_drop_out = opt.rnn_dropout_out

        # ODE Term and Integrator
        self.ode_func = ODEFunc(
            feature_dim=self.f_len,
            hidden_dim=opt.ode_hidden_dim,
            num_hidden_layers=opt.ode_fn_num_layers,
            activation=opt.ode_activation_fn,
        )
        term = to.ODETerm(self.ode_func)
        step_method = self._set_solver(opt.ode_solver, term=term)
        step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-2, term=term)
        # step_size_controller = to.FixedStepController()
        self.solver = torch.compile(
            to.AutoDiffAdjoint(step_method, step_size_controller)
        )
        # self.update_solver(step_size_controller, step_method)
        # print("Tolerance:", step_size_controller.atol, step_size_controller.rtol)
        # print(self.solver)
        self.rnn = self._set_rnn(opt.ode_rnn_type)
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        self.fuse = FusionModule(feature_dim=self.f_len, fuse_method=self.fuse_method)
        self.regressor = nn.Sequential(
            nn.Linear(self.f_len, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6),
        )
    
    def evolve_state(self, state, ts):
        # This function will handle the ODE evolution for a single hidden state
        dt0 = torch.full((ts.shape[0],), 0.0001, device=ts.device)
        problem = to.InitialValueProblem(y0=state, t_eval=ts)
        ode_solution = self.solver.solve(problem, dt0=dt0)
        return ode_solution.ys[:, -1, :]

    def update_method(self):
        term = to.ODETerm(self.ode_func)
        step_method = to.Euler(term=term)
        # step_size_controller = to.IntegralController(atol=1e-1, rtol=1e1, term=term)
        # step_size_controller = to.FixedStepController() 
        # print("Tolerance:", step_size_controller.atol, step_size_controller.rtol) 
        step_size_controller = to.FixedStepController() 
        self.solver = torch.compile(
            to.AutoDiffAdjoint(step_method, step_size_controller)
        )
        print("updated_method")
        
    def forward(self, fv, fi, ts, prev=None, do_profile=False):
        
        # self.update_method()
        # Fuse visual and inertial features
        fused_features = self.fuse(fv, fi)
        batch_size, seq_len, _ = fused_features.shape
        
        # Initialise initial state
        h_0 = torch.zeros(self.rnn_num_layers, batch_size, self.f_len, device=fused_features.device) if prev is None else prev
        
        # Subtract the first timestamp from all timestamps to get time differences
        # Remove the first timestamp and add a dimension
        # TODO: Evalute whether to reset during testing, where the seqeunce is continuous
        ts_diff = ts - ts[:, :1] if prev is None else ts
        
        # Profiling 
        if do_profile:
            torch.cuda.nvtx.range_push("odeint")
        
        output = []
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # total_time = 0
        # Evolve the state using the ODE solver
        for i in range(seq_len):
            # start.record()
            futures = [torch.jit.fork(self.evolve_state, h_0[j], ts_diff[:, i:i+2]) for j in range(self.rnn_num_layers)]
            ode_hidden_states = [torch.jit.wait(future) for future in futures]
            ode_hidden_states = torch.stack(ode_hidden_states, dim=0)
            # end.record()
            # torch.cuda.synchronize()
            # elapsed_time_ms = start.elapsed_time(end)
            # print(f'Elapsed time: {elapsed_time_ms:.3f} ms')
            # total_time += elapsed_time_ms
            # Pass the fused features and the evolved hidden states through the RNN
            output_i, rnn_h = self.rnn(fused_features[:, i : i + 1, :], ode_hidden_states)
            output.append(output_i)
            h_0 = rnn_h
        output = torch.cat(output, dim=1)
        # print(total_time/10)
        if do_profile:
            torch.cuda.nvtx.range_pop()
        
        # Regress the network to get the pose
        pose = self.regressor(output)
        return pose, h_0

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
        return solver

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