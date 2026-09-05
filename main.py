from models.model import Model
from models.train import Training
from data.generate_data import generate_block
from models.scheduler import cosine_warmup

import torch.nn
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    steps = 10000

    model = Model().to(device)
    
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    schedular3 = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=steps,
        eta_min=0.0,
        last_epoch=-1
    )

    schedular = cosine_warmup(
        optimizer=optimizer,
        warmup_steps=300,
        total_steps=steps
    )

    #loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.HuberLoss()

    trainer = Training(
        optimizer=optimizer, 
        loss_fn=loss_fn, 
        schedular=schedular,
        device=device,
        data_gen=generate_block, 
        model=model
    )
    
    trainer.train(steps=steps, timesteps=20)
