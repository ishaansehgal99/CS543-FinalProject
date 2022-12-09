import torch
import torch.nn.functional as F
import torch.nn as nn
from models import Generator
import matplotlib.pyplot as plt


class LSTM_Model(torch.nn.Module):
    def __init__(self,noise_dim, state_dict):
        super().__init__()
        self.Generator = Generator(noise_dim)
        self.Generator.load_state_dict(torch.load(state_dict))
        self.noise_dim = noise_dim
        self.lstm = nn.LSTM(529040, noise_dim, batch_first=True)
        
    def forward(self, x):
        x = x.view(-1, self.noise_dim)
        x.unsqueeze(-1)
        x, _ = self.lstm(x)

        img = self.Generator(x)
        return img
    
    
    
    def main():
        
        z = torch.rand((2, 40, 6613))
        # z = z.flatten()
        z = z.unsqueeze(dim = 0)
        g = Generator(529040)
        out = g(z)
        print(out.size())  
        