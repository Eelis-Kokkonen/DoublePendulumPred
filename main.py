from models.model import Model
from models.training import Training
from data.generate_data import generate_block

import torch.nn
import torch

if __name__ == "__main__":

    optimizer = nn.AdamW()

    loss_fn = 
    
    schedular = 


    trainer = Trainer(
        optimizer=optimizer, 
        loss_fn=loss_fn, 
        schedular=schedular,
        data_gen=generate_block, 
        model=Model
    )
    
    
    
    trainer.train(steps=1_000)


                 optimizer, 
                 loss_fn, 
                 shcedular,
                 device,
                 data_gen=generate_block, 
                 model=Model

