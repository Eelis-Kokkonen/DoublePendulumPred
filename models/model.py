import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, 
                 state_dim=4,
                 param_dim=5,
                 d_model=64, 
                 d_ff=128, 
                 num_layers=4,
                 nhead=2
                ):

        super().__init__()

        self.input_proj = nn.Linear(state_dim, d_model)
        self.param_proj = nn.Linear(param_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.output_proj = nn.Linear(d_model, state_dim)  

    def forward(self, x, params):

        x_emb = self.input_proj(x)

        p_emb = self.param_proj(params).unsqueeze(1)

        tokens = torch.cat([p_emb, x_emb], dim=1)

        out = self.transformer(tokens)

        next_state = self.output_proj(out[:, -1, :])
        
        return next_state


    def predict(self, x, params, timesteps=1_000):

        predictions = []

        curr_x = x.clone()

        for _ in range(timesteps):
            delta = self.forward(curr_x, params)

            next_state = curr_x[:, -1, :] + delta
            
            next_state_seq = next_state.unsqueeze(1)

            predictions.append(next_state_seq)

            curr_x = torch.cat([curr_x, next_state_seq], dim=1)

        return torch.cat(predictions, dim=1)
