import torch
import numpy as np
from models import *
from LSTM import LSTM_Model
from models import *
import torch.nn as nn
from dataset import *
from torch.utils.data import DataLoader
import tqdm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from album_data import *

class LSTM_trainer():
    def __init__(self, device, lr, state_dict, batch_size = 8):
        self.LSTM = LSTM_Model(128,state_dict).to(device)
        self.device = device
        self.batch_size = batch_size
        self.optimL = torch.optim.Adam(self.LSTM.lstm.parameters(),lr)
        self.dataset = SpotifyData("./spotify_data/song_wavs","./spotify_data/album_covers",True)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
    def train(self, num_epochs):
        loss_fn = nn.L1Loss()
        for e in range(1, num_epochs + 1):
            loop = tqdm.tqdm(self.dataloader, leave=True)
            epoch_loss = 0
            print("Epoch: ", e)
            for idx, (img,sound) in enumerate(loop):
                
                self.optimL.zero_grad()
                out = self.LSTM(sound)
                loss = loss_fn(out, img)
                loss.backward()
                epoch_loss += loss.item()
                self.optimL.step()
                
            if e % 5 == 0:
                print("Epoch: ", e)
                #print("Iter: ", idx)
                torch.save(self.LSTM.state_dict(), "curr_lstm.pth")
                
            
            print("Epoch loss LSTM: ", epoch_loss)
            torch.save(self.LSTM.state_dict(), "curr_lstm.pth")

