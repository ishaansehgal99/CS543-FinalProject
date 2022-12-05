import torch
from torch.utils.data.dataloader import Dataset
from os import listdir
from os.path import isfile, join
from torchvision.transforms import transforms
from PIL import Image
import numpy as np




class AlbumData(Dataset):
    def __init__(self, imagesPath):
        super().__init__()
        self.images = []
        self.loadData(imagesPath)
    

    def loadData(self, imagesPath):
        imageFiles = [imagesPath + "/" + f for f in listdir(imagesPath) if isfile(join(imagesPath, f))]
        self.images = imageFiles
        print("Loaded Images")


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        trans = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64))])
        image = Image.open(self.images[index])
        image = trans(image)
        return image


def main():
    data = AlbumData("./spotify_data/archive/album_covers_512")
    print(len(data))


if __name__ == "__main__":
    main()
