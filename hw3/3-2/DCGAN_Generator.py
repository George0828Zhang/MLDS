import torch
import torch.nn as nn
import numpy as np
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class MyGenerator(torch.nn.Module):
    def __init__(self, img_shape, latent_dim, text_dim):
        super(MyGenerator, self).__init__()
        self.hidden_dim = 128
        self.img_shape = img_shape;   
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        
        #self.Relu = nn.ReLU(True);
        #self.linear = torch.nn.Linear(latent_dim, self.hidden_dim* 4* 4)
        
        self.embed = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(True)
        )
        
        self.mlp = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( latent_dim + text_dim, self.hidden_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.hidden_dim * 8),
            nn.ReLU(True),
            # state size. (self.hidden_dim*8) x 4 x 4
            nn.ConvTranspose2d(self.hidden_dim * 8, self.hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim * 4),
            nn.ReLU(True),
            # state size. (self.hidden_dim*4) x 8 x 8
            nn.ConvTranspose2d( self.hidden_dim * 4, self.hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim * 2),
            nn.ReLU(True),
            # state size. (self.hidden_dim*2) x 16 x 16
            nn.ConvTranspose2d( self.hidden_dim * 2, self.hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(True),
            # state size. (self.hidden_dim) x 32 x 32
            nn.ConvTranspose2d( self.hidden_dim, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (3) x 64 x 64
        )


    def forward(self, x, txt):
        #x = self.linear(x)        
        #x = self.Relu(x).view(x.shape[0],128,4,4)
        batch_size = x.shape[0]
        
        #expect (batch, 119)
        emb = self.embed(txt)
        print("[debug][gen]",emb.shape)
        #expect (batch, 256)
        
        x = x.reshape(-1, self.latent_dim, 1, 1);
        emb = emb.view(batch_size, self.text_dim, 1, 1)
        
        x = torch.cat((x, emb), 1)
        
        x = self.mlp(x)
        return x.transpose(2,1).transpose(3,2)
