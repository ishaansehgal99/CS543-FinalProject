import torch
from torch.utils.data.dataloader import Dataset
import librosa
from os import listdir
from os.path import isfile, join
from torchvision.transforms import transforms
from PIL import Image
import numpy as np




class SpotifyData(Dataset):
    def __init__(self, soundsPath, imagesPath, load_sounds):
        super().__init__()
        self.images = []
        self.sounds = []
        self.load_sounds = load_sounds
        self.loadData(soundsPath, imagesPath)
    

    def loadData(self, soundsPath, imagesPath):
        soundFiles = [soundsPath + "/" + f for f in listdir(soundsPath) if isfile(join(soundsPath, f))]
        soundsWithoutF = [f for f in listdir(soundsPath) if isfile(join(soundsPath, f))]
        trans = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64))])
        audioTrans = transforms.Compose([transforms.ToTensor()])
        dim_wanted = torch.empty((1, 20, 1292))
        MAX = 10000
        for i in range(len(soundFiles)):
            if i - 1 == MAX:
                break

            sound = soundsWithoutF[i]
            imageName = sound.split(".")[0]
            if self.load_sounds:
                data, sr = librosa.load(soundFiles[i])
                audio = librosa.feature.mfcc(y = data, sr = sr)
                # print(audio.shape)
                audio = audioTrans(audio)
                if audio.size() != dim_wanted.size():
                    continue
                audio = audio.view(25840)
                self.sounds.append(audio)

            try:
                image = Image.open(imagesPath + "/" + imageName + ".jpeg")
                image = np.array(image)
                image = trans(image)
                s = torch.empty((3, 64, 64))
                if image.size() != s.size():
                    raise Exception
            except:
                continue

            self.images.append(image)
        print(len(self.images))
        print("Loaded Sounds & Images")


    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        if self.load_sounds:
            return self.images[index], self.sounds[index]
        else:
            return self.images[index]


def main():
    data = SpotifyData("./spotify_data/song_wavs", "./spotify_data/album_covers")
    print(len(data))


if __name__ == "__main__":
    main()
