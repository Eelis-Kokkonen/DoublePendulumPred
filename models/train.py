from models.model import Model
from data.generate_data import generate_block
from physics.simulator import generate_states


import torch
from tqdm import tqdm

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

        self.eval_freq = 1_000

        self.init_state = 5

        self.num_sims = 20
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

            loss = self.loss_fn(pred, eval_traj)

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

                loss = self.loss_fn(pred, eval_traj)

                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "schedular_state_dict": self.schedular.state_dict() if self.schedular else None,
                    "steps": step,
                    "loss": loss
                }, f"checkpoint_{step}.pth")

        print("Training has ended...")
        
