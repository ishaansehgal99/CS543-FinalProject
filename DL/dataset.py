import torch
from torch.utils.data.dataloader import Dataset
import torchaudio
from os import listdir
from os.path import isfile, join
import random
from torchvision.transforms import transforms
from PIL import Image
import numpy as np




class SpotifyData(Dataset):
    def __init__(self, soundsPath, imagesPath):
        super().__init__()
        self.images = []
        self.sounds = []
        self.loadData(soundsPath, imagesPath)
    

    def loadData(self, soundsPath, imagesPath):
        soundFiles = [soundsPath + "/" + f for f in listdir(soundsPath) if isfile(join(soundsPath, f))]
        soundsWithoutF = [f for f in listdir(soundsPath) if isfile(join(soundsPath, f))]
        trans = transforms.Compose([transforms.ToTensor(), transforms.Resize(64)])
        dim_wanted = torch.empty((2, 40, 6613))

        for i in range(len(soundFiles)):
            sound = soundsWithoutF[i]
            imageName = sound.split(".")[0]
            audio, sample_rate = torchaudio.load(soundFiles[i])
            mfccTrans = torchaudio.transforms.MFCC(sample_rate)
            audio = mfccTrans(audio)
            if audio.size() != dim_wanted.size():
                continue
            
            image = Image.open(imagesPath + "/" + imageName + ".jpeg")
            self.sounds.append(audio)
            image = np.array(image)
            image = trans(image)
            self.images.append(image)
        print(len(self.images))
        print("Loaded Sounds & Images")


    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.sounds[index]


def main():
    data = SpotifyData("./spotify_data/song_clips", "./spotify_data/album_covers")
    print(len(data))


if __name__ == "__main__":
    main()
