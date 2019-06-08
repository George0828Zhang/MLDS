import torch
import torch.nn as nn
import numpy as np
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)    
    
class MyGenerator(torch.nn.Module):
    def __init__(self, img_shape, latent_dim, text_dim):
        super(MyGenerator, self).__init__()              
        #self.hidden_dim = 64
        self.img_shape = img_shape;   
        self.latent_dim = latent_dim
        self.text_dim = text_dim
       
        self.fc1 = nn.Linear(latent_dim + text_dim, 384)
        
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(384, 256, 5, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 5, 2, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 5, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        #output [-1, 64, 61, 61] 
        
        #(61-1) * 1 + 0 + 0 + 4
        # Transposed Convolution 5
        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, 1, 0, bias=False),
            nn.Tanh(),
        )                  
        #self.apply(weights_init)

        
    def forward(self, x, txt): 
        batch_size = x.shape[0]
        x = torch.cat([x,txt], 1)
        x = x.view(-1, self.latent_dim + self.text_dim)
        fc1 = self.fc1(x)
        fc1 = fc1.view(-1, 384, 1, 1)
        tconv2 = self.tconv2(fc1)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        output = self.tconv6(tconv5)

        return output.transpose(2,1).transpose(3,2)
