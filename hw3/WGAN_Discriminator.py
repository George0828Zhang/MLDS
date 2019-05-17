import torch
import torch.nn as nn
import numpy as np
class MyDiscriminator(torch.nn.Module):
    def __init__(self, img_shape):
        super(MyDiscriminator, self).__init__()
        self.hidden_dim = 32
        #int(np.prod(img_shape))
        self.n_channel = 3;
        
        self.mlp = nn.Sequential(
            # input is (n_channel) x 64 x 64
            nn.Conv2d(self.n_channel, self.hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.hidden_dim) x 32 x 32
            nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.hidden_dim*2) x 16 x 16
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.hidden_dim*4) x 8 x 8
            nn.Conv2d(self.hidden_dim * 4, self.hidden_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.hidden_dim*8) x 4 x 4
            nn.Conv2d(self.hidden_dim * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )
        
    def forward(self, img):
        #print("dis")
        if(img.shape[1] != 3):
            img = img.transpose(3,2).transpose(2,1)
        out = self.mlp(img)
        return out.squeeze(-1)  