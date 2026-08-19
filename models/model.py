import torch
import torch.nn
import torch.nn.Functional as F


class Model(nn.Module):
    def __init__(self, 
                 input_dim=16, 
                 d_model=64, 
                 d_ff=128, 
                 num_layers=4, 
                 output_dim=12):

        self.latent = nn.Parameter(torch.randn(1, 1, d_model))

        self.input = nn.Linear(input_dim, d_model)

        self.attention = Transformer(d_model=d_model, 
                                     n_head=2, 
                                     num_encoder_layers=0, 
                                     num_decoder_layers=num_layers, 
                                     dim_feedforward=d_ff, 
                                     activation=F.SiLU, 
                                     batch_first=True)

        self.output = nn.Linear(d_model, output_dim)  

    def forward(self, x):

        B = x.shape(0)
      
        x = self.input(x)

        latent = self.latent.expand(B, -1, -1)

        x = torch.cat(latent, x)
      
        x = self.attention(x)

        x = x[:, 0, :]
      
        x = self.output(x)
        
        return x


    def predict(self, x, timesteps=1_000):
        
        B = x.shape(0)

        M = x.extract(:, :, 13:14)

        G = x.extract(:, :, 15)

        for i in range(timesteps):

            pred = self.forward(x)

            x = 





        return predictions
