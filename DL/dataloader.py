import torch
import torchaudio
from torch.utils.data.dataloader import Dataset
import torch.nn as nn
from os import listdir
from os.path import isfile, join



class DataLoader(Dataset):
    def __init__(self, soundsPath, imagesPath):
        super().__init__()
        self.images = []
        self.sounds = []
        self.loadData(soundsPath, imagesPath)
    

    def loadData(self, soundsPath, imagesPath):
        onlyfiles = [f for f in listdir(soundsPath) if isfile(join(soundsPath, f))]
        print(onlyfiles)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.sounds[index]

    
