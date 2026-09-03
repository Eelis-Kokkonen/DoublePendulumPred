from model import Model
from data.generate_data import generate_block

import torch
from tqdm import tqdm

class Training:
    def __init__(self, 
                 optimizer, 
                 loss_fn, 
                 shcedular,
                 device,
                 data_gen=generate_block, 
                 model=Model
                ):

        self.device = device
        self.model = model

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.schedular = schedular
                    
        self.data_gen = data_gen

        self.eval_freq = 1_000

    def train(self, steps=1_000):
        
        print("Training has started...")

        pbar = tqdm(range(steps))

        for step in pbar:
            model.train()

            self.optimizer.zero_grad()

            traj = data_gen()

            train_traj
            eval_traj

            pred = model.predict(train_traj, timesteps=1_000)

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
                
                traj_eval = data_gen()

                input_traj
                pred_traj

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
        
