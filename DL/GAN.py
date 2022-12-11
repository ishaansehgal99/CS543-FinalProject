import torch
import numpy as np
from sagan_models import *
import torch.nn as nn
from dataset import *
from torch.utils.data import DataLoader
import tqdm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from album_data import *
import numpy as np



class GAN():
    def __init__(self, device, lr, batch_size = 16):
        self.G = Generator(batch_size=batch_size).to(device)
        self.D = Discriminator(batch_size=batch_size).to(device)
        self.device = device
        self.batch_size = batch_size

        self.optimG = torch.optim.Adam(self.G.parameters(), 0.0001, betas=(0.0, 0.99))
        self.optimD = torch.optim.Adam(self.D.parameters(), 0.0004, betas=(0.0, 0.99))

        self.dataset = AlbumData("./spotify_data/archive/album_covers_512")
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        #self.weights_init(self.G)
        #self.weights_init(self.D)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train(self, num_epochs):
        loss_fn = nn.MSELoss()
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
                z = torch.randn((self.batch_size, 100)).to(self.device)
                z /= torch.max(z)
                fake_img, _, _= self.G(z)
                fake_img = fake_img.detach()
                fake_out, _, _ = self.D(fake_img)
                real_out, _, _ = self.D(img)


                loss = loss_fn(real_out, torch.ones_like(real_out, device=self.device)) + loss_fn(fake_out, torch.zeros_like(fake_out, device=self.device))
                loss /= 2.0
                epoch_loss_D += loss.item()
                loss.backward()
                self.optimD.step()


                # Train G
                # z = torch.randn((self.batch_size, 100)).to(self.device)
                # z /= torch.max(z)
                self.optimG.zero_grad()
                fake_img, _, _ = self.G(z)
                fake_out, _, _ = self.D(fake_img)
                loss = loss_fn(fake_out, torch.ones_like(fake_out, device=self.device)) / 2.0
                epoch_loss_G += loss.item()
                loss.backward()
                self.optimG.step()

                if idx % 2000 == 0:
                    fake_img = fake_img.cpu().detach()
                    grid = make_grid(fake_img[:4], 4)
                    plt.imshow(grid.permute(1, 2, 0))
                    plt.show()


            if e % 1 == 0:
                print("Epoch: ", e)
                #print("Iter: ", idx)
                fake_img = fake_img.cpu().detach()
                grid = make_grid(fake_img[:4], 4)
                plt.imshow(grid.permute(1, 2, 0))
                plt.show()
            
            print("Epoch loss G: ", epoch_loss_G)
            print("Epoch loss D: ", epoch_loss_D)
            torch.save(self.G.state_dict(), "SA5curr_gen.pth")
            torch.save(self.D.state_dict(), "SA5curr_dis.pth")



    








