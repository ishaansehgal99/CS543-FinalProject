import torch
import numpy as np
from models import *
import torch.nn as nn
from dataset import *
from torch.utils.data import DataLoader
import tqdm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from album_data import *



class GAN():
    def __init__(self, device, lr, batch_size = 8):
        self.G = Generator(128).to(device)
        self.D = Discriminator().to(device)
        self.device = device
        self.batch_size = batch_size

        self.optimG = torch.optim.Adam(self.G.parameters(), lr)
        self.optimD = torch.optim.Adam(self.D.parameters(), lr)

        self.dataset = AlbumData("./spotify_data/archive/album_covers_512")
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    

    def train(self, num_epochs):
        for e in range(1, num_epochs + 1):
            loop = tqdm.tqdm(self.dataloader, leave=True)
            epoch_loss_G = 0
            epoch_loss_D = 0
            print("Epoch: ", e)
            for idx, img in enumerate(loop):
                # Train D
                self.optimD.zero_grad()
                img = img.to(self.device)
                #sound = sound.to(self.device)

                #Generate fake images
                z = torch.randn((self.batch_size, 128)).to(self.device)
                fake_img = self.G(z).detach()
                fake_out = self.D(fake_img)
                real_out = self.D(img)


                loss = (torch.mean(torch.pow(real_out - 1.0, 2)) + torch.mean(torch.pow(fake_out, 2))) / 2.0
                epoch_loss_D += loss.item()
                loss.backward()
                self.optimD.step()


                # Train G
                z = torch.randn((self.batch_size, 128)).to(self.device)
                self.optimG.zero_grad()
                fake_img = self.G(z)
                fake_out = self.D(fake_img)
                loss = torch.mean(torch.pow(fake_out - 1.0, 2)) / 2.0
                epoch_loss_G += loss.item()
                loss.backward()
                self.optimG.step()

            if e % 5 == 0:
                print("Epoch: ", e)
                #print("Iter: ", idx)
                torch.save(self.G.state_dict(), "curr_gen.pth")
                torch.save(self.D.state_dict(), "curr_dis.pth")

                fake_img = fake_img.cpu().detach()
                grid = make_grid(fake_img, self.batch_size)
                plt.imshow(grid.permute(1, 2, 0))
                plt.show()
            
            print("Epoch loss G: ", epoch_loss_G)
            print("Epoch loss D: ", epoch_loss_D)
            torch.save(self.G.state_dict(), "curr_gen.pth")
            torch.save(self.D.state_dict(), "curr_dis.pth")



    








