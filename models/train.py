from models.model import Model
from data.generate_data import generate_block
from physics.simulator import generate_states


import torch
from tqdm import tqdm



import matplotlib.pyplot as plt
import numpy as np
import torch

def compute_step_transition_metrics(pred, target):
    """
    Computes step-to-step transition errors between adjacent time steps (tokens).
    
    Args:
        pred: Tensor of shape (batch_size, pred_len, state_dim)
        target: Tensor of shape (batch_size, pred_len, state_dim)
        
    Returns:
        transition_error: MSE of predicted state changes (step t -> t+1)
        error_growth_rate: Incremental increase in cumulative loss (Error(t+1) - Error(t))
    """
    # 1. Transition Error: Error in state delta between token t and token t+1
    pred_delta = pred[:, 1:, :] - pred[:, :-1, :]
    target_delta = target[:, 1:, :] - target[:, :-1, :]
    
    transition_error = torch.mean((pred_delta - target_delta) ** 2, dim=(0, 2)).detach().cpu().numpy()
    
    # 2. Cumulative MSE per timestep
    step_mse = torch.mean((pred - target) ** 2, dim=(0, 2)).detach().cpu().numpy()
    
    # 3. Incremental Error Growth: Loss added strictly at token t+1 compared to token t
    error_growth_rate = np.diff(step_mse)
    
    return transition_error, error_growth_rate

def plot_token_step_errors(transition_error, error_growth_rate, step, filename=None):
    """Plots transition delta error and step-by-step error growth across token sequence."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    steps = np.arange(1, len(transition_error) + 1)
    
    # Plot 1: Transition Delta Error between consecutive tokens
    ax1.plot(steps, transition_error, color="darkorange", linewidth=1.5, label=r"Transition Error ($\Delta t \to \Delta t+1$)")
    ax1.set_yscale("log")
    ax1.set_ylabel("Transition MSE (Log Scale)")
    ax1.set_title(f"Token-to-Token Step Dynamics Error (Training Step {step})")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend()
    
    # Plot 2: Incremental loss growth added per timestep
    ax2.plot(steps, error_growth_rate, color="purple", linewidth=1.5, label=r"Loss Acceleration ($\text{MSE}_{t+1} - \text{MSE}_t$)")
    ax2.set_xlabel("Token Transition Index ($t \to t+1$)")
    ax2.set_ylabel("Incremental Loss Change")
    ax2.set_title("Per-Token Error Acceleration Rate")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    save_path = filename if filename else f"token_step_error_step_{step}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()


class Training:
    def __init__(self, 
                 optimizer, 
                 loss_fn, 
                 schedular,
                 device,
                 data_gen=generate_block, 
                 model=Model
                ):

        super().__init__()
                    
        self.device = device
        self.model = model

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.schedular = schedular
                    
        self.data_gen = data_gen

        self.eval_freq = 100_000

        self.init_state = 5

        self.num_sims = 32
        self.dt = 0.01

    def train(self, steps=1_000, timesteps=1_000):

        pred_len = timesteps - self.init_state
        
        print("Training has started...")

        pbar = tqdm(range(steps))

        for step in pbar:
            self.model.train()

            self.optimizer.zero_grad()

            initial_states = generate_states(self.num_sims)

            params = torch.from_numpy(initial_states[:, 4:]).to(self.device, dtype=torch.float32)

            traj = torch.from_numpy(self.data_gen(initial_states, self.dt, timesteps)).to(self.device, dtype=torch.float32)
            
            train_traj = traj[:, :self.init_state, :]
            eval_traj = traj[:, self.init_state:, :]

            pred = self.model.predict(train_traj, params, timesteps=pred_len)

            trans_err, growth_rate = compute_step_transition_metrics(pred_eval, target_traj_eval)
            plot_token_step_errors(trans_err, growth_rate, step=step + 1)

            loss = self.loss_fn(pred, eval_traj)

            loss = torch.clamp(loss, max=200.0)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            pbar.set_postfix(
                loss=f"{loss.item():.10f}"
            )

            if self.schedular is not None:
                self.schedular.step()

            if (step + 1)  % self.eval_freq == 0:
                self.model.eval()

                initial_states_eval = generate_states(self.num_sims)

                params_eval = torch.from_numpy(initial_states_eval[:, 4:]).to(self.device, dtype=torch.float32)
                
                traj_eval = torch.from_numpy(self.data_gen(initial_states_eval, self.dt, timesteps)).to(self.device, dtype=torch.float32)

                input_traj = traj_eval[:, :pred_len, :]
                pred_traj = traj_eval[:, pred_len:, :]

                pred_eval = self.model.predict(input_traj, params_eval, timesteps=1_000)

                loss = self.loss_fn(pred_eval, eval_traj)

                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "schedular_state_dict": self.schedular.state_dict() if self.schedular else None,
                    "steps": step,
                    "loss": loss
                }, f"checkpoint_{step}.pth")

        print("Training has ended...")
        
