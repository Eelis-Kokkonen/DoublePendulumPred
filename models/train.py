from models.model import Model
from data.generate_data import generate_block
from physics.simulator import generate_states

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

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


def save_comparison_gif(
    target_traj, pred_traj, filename, sample_idx=0, fps=30, l1=1.0, l2=1.0
):
    """Generates and saves a stacked GIF animating a Double Pendulum system.

    Top: Ground Truth Physics Simulator.
    Bottom: Model Prediction.

    Args:
        target_traj: Tensor/ndarray of shape (batch_size, timesteps, 4) [theta1, theta2, omega1, omega2]
        pred_traj: Tensor/ndarray of shape (batch_size, timesteps, 4)
        filename: Path to save output .gif
        sample_idx: Index of sample in batch to render
        fps: Frames per second
        l1: Length of inner rod 1
        l2: Length of outer rod 2
    """
    if torch.is_tensor(target_traj):
        target_traj = target_traj.detach().cpu().numpy()
    if torch.is_tensor(pred_traj):
        pred_traj = pred_traj.detach().cpu().numpy()

    gt = target_traj[sample_idx]
    pred = pred_traj[sample_idx]
    num_frames = min(len(gt), len(pred))

    # Compute double pendulum bob locations via forward kinematics
    def get_double_pendulum_coords(seq):
        theta1, theta2 = seq[:, 0], seq[:, 1]
        
        # Bob 1 coordinates
        x1 = l1 * np.sin(theta1)
        y1 = -l1 * np.cos(theta1)
        
        # Bob 2 coordinates
        x2 = x1 + l2 * np.sin(theta2)
        y2 = y1 - l2 * np.cos(theta2)
        
        return x1, y1, x2, y2

    x1_gt, y1_gt, x2_gt, y2_gt = get_double_pendulum_coords(gt)
    x1_pred, y1_pred, x2_pred, y2_pred = get_double_pendulum_coords(pred)

    fig, (ax_gt, ax_pred) = plt.subplots(2, 1, figsize=(6, 8))

    limit = (l1 + l2) * 1.25

    for ax, title in zip([ax_gt, ax_pred], ["Ground Truth Physics Simulation", "Model Prediction"]):
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        # Fixed base pivot
        ax.plot(0, 0, "ko", markersize=7, zorder=5)

    # Initialize Ground Truth double pendulum elements
    (rods_gt,) = ax_gt.plot([], [], "o-", color="navy", lw=2.5, markersize=8, zorder=4)
    (trace_gt,) = ax_gt.plot([], [], "--", color="blue", alpha=0.35, lw=1.5, zorder=2)

    # Initialize Model Prediction double pendulum elements
    (rods_pred,) = ax_pred.plot([], [], "o-", color="darkred", lw=2.5, markersize=8, zorder=4)
    (trace_pred,) = ax_pred.plot([], [], "--", color="red", alpha=0.35, lw=1.5, zorder=2)

    def init():
        rods_gt.set_data([], [])
        trace_gt.set_data([], [])
        rods_pred.set_data([], [])
        trace_pred.set_data([], [])
        return rods_gt, trace_gt, rods_pred, trace_pred

    def update(frame):
        # 1. Update Ground Truth (Base -> Bob 1 -> Bob 2)
        rods_gt.set_data([0, x1_gt[frame], x2_gt[frame]], [0, y1_gt[frame], y2_gt[frame]])
        
        # Draw motion trail of tip (Bob 2) for the last 30 steps
        trail_start = max(0, frame - 30)
        trace_gt.set_data(x2_gt[trail_start : frame + 1], y2_gt[trail_start : frame + 1])

        # 2. Update Model Prediction (Base -> Bob 1 -> Bob 2)
        rods_pred.set_data([0, x1_pred[frame], x2_pred[frame]], [0, y1_pred[frame], y2_pred[frame]])
        trace_pred.set_data(x2_pred[trail_start : frame + 1], y2_pred[trail_start : frame + 1])

        return rods_gt, trace_gt, rods_pred, trace_pred

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=num_frames, interval=1000 // fps, blit=True
    )

    plt.tight_layout()
    anim.save(filename, writer="pillow", fps=fps)
    plt.close(fig)



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

            if (step + 1) % 100 == 0:
                trans_err, growth_rate = compute_step_transition_metrics(pred, eval_traj)
                plot_token_step_errors(trans_err, growth_rate, step=step + 1)

                # Extract rod lengths for visualization sample index 0
                sample_l1 = float(initial_states[0, 4])
                sample_l2 = float(initial_states[0, 5])

                save_comparison_gif(
                    target_traj=eval_traj,
                    pred_traj=pred,
                    filename=f"simulation_vs_model_step_{step+1}.gif",
                    sample_idx=0,
                    fps=20,
                    l1=sample_l1,
                    l2=sample_l2
                )

            loss = self.loss_fn(pred, eval_traj)

            #loss = torch.clamp(loss, max=00.0)

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
        
