import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, mac_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

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
            dropout=0,
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

        seq_len = tokens.size(1)
        
        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool
            ), diagonal=1)
        
        out = self.transformer(tokens, mask)

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



class Model2(nn.Module):
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
        self.param_proj = nn.Linear(param_dim+state_dim, d_model)

        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=1_000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=0,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.output_proj = nn.Linear(d_model, state_dim)  

    def forward(self, x, params):

        x0 = x[:, 0, :]

        x_rest = x[:, 1:, :]

        x_emb = self.input_proj(x_rest)

        params_input = torch.cat([x0, params], dim=1)

        p_emb = self.param_proj(params).unsqueeze(1)

        tokens = torch.cat([p_emb, x_emb], dim=1)

        tokens = self.pos_encoder(tokens)

        seq_len = tokens.size(1)
        
        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool
            ), diagonal=1)
        
        out = self.transformer(tokens, mask)

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



class Model3(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, output_dim=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            
            nn.Linear(hidden_dim, output_dim),
            
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, initial_state, params, timesteps=1_000):
        predictions = []

        curr_state = initial_state[:, -1, :] if initial_state.ndim == 3 else initial_state

        for i in range(timesteps):
            x = torch.cat([curr_state, params], dim=-1)
            delta = self.forward(x)
            curr_state = curr_state + delta
            predictions.append(curr_state.unsqueeze(1))

        return torch.cat(predictions, dim=1)

