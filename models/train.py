from models.model import Model
from data.generate_data import generate_block

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

        self.init_state = 100

    def train(self, steps=1_000, timesteps=1_000):

        pred_len = timesteps - init_state
        
        print("Training has started...")

        pbar = tqdm(range(steps))

        for step in pbar:
            model.train()

            self.optimizer.zero_grad()

            traj = torch.from_numpy(data_gen())

            train_traj = traj[:, :self.init_state, :]
            eval_traj = traj[:, self.init_state:, :]

            pred = model.predict(train_traj, timesteps=pred_len)

            loss = self.loss_fn(pred, eval_traj)

            loss.backwards()

            self.optimizer.step()

            pbar.set_postfix(
                loss=f"{loss.item():.f3}"
            )

            if schedular is not None:
                self.schedular.step()

            if (step + 1)  % eval_freq == 0:
                model.eval()
                
                traj_eval = torch.from_numpy(data_gen())

                input_traj = traj_eval[:, :pred_len, :]
                pred_traj = traj_eval[:, pred_len:, :]

                pred_eval = model.predict(input_traj, timesteps=1_000)

                loss = self.loss_fn(pred, eval_traj)

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "schedular_state_dict": schedular.state_dict() if schedular else None,
                    "steps": step,
                    "loss": loss
                }, f"checkpoint_{step}.pth")

        print("Training has ended...")
        
