import torch
import torch.nn.functional as F
import torch.nn as nn
class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        

        
        self.conv1 =torch.nn.Conv2d(3, 128, kernel_size = 4, stride= 2, padding =1)
        self.conv2 = torch.nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding=1)
        self.bn1 =torch.nn.BatchNorm2d(256)
        self.conv3 = torch.nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(512)
        self.conv4 = torch.nn.Conv2d(512, 1024, kernel_size = 4, stride = 2, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(1024)
        self.conv5 = torch.nn.Conv2d(1024, 1, kernel_size = 4, stride = 1, padding = 1)
        self.leaky_relu = torch.nn.LeakyReLU(0.2) 
         
        
    def forward(self, x):
        
    
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.bn1(self.conv2(x))) 
        x = self.leaky_relu(self.bn2(self.conv3(x)))
        x = self.leaky_relu(self.bn3(self.conv4(x)))
        x = self.conv5(x)
        
        
        return x


class Generator(nn.Module):
    
    def __init__(self, noise_dim):
        super().__init__()

        self.noise_dim = noise_dim
    
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(self.noise_dim, 1024, 4, 1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 3, 4, 2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = x.view(-1, self.noise_dim, 1, 1)
        return self.gen(x)
    

def main():
    # g = Generator(529040)
    z = torch.rand((2, 40, 6613))
    # z = z.flatten()
    z = z.unsqueeze(dim = 0)
    
    # print(z.size())
    # out = g(z)
    # print(out.size())
    # z = torch.rand((1, 529040))
    # lstm = nn.LSTM(529040, 1000 , batch_first=True)
    # ho = torch.zeros((1, 1000))
    # co = torch.zeros((1, 1000))
    # o1, h = lstm(z,(ho, co))
    # print(o1.size())









if "__main__" == __name__:
    main()