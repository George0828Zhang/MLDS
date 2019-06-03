import torch
import torch.nn as nn
import numpy as np
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class MyDiscriminator(torch.nn.Module):
    def __init__(self, img_shape, text_dim):
        
        super(MyDiscriminator, self).__init__()
                
        self.hidden_dim = 32
        self.n_channel = 3
        self.text_dim = text_dim
        
        self.conv_layers = nn.Sequential(
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
            # state size. (self.hidden_dim*8) x 8 x 8
            
            #nn.Sigmoid(),

        )
        
        self.score_layer =  nn.Sequential(
            #nn.Conv2d(self.hidden_dim * 8, 1, 2, 1, 0, bias=False),
            nn.Linear(4096 , 1),
            #nn.Sigmoid(),
        )
        
        

        self.classifier = nn.Sequential(
            nn.Linear(4096, text_dim),            
        )
        
        self.apply(weights_init)
        
    def forward(self, img):
        
        if(img.shape[1] != 3):
            img = img.transpose(3,2).transpose(2,1)
        
        batch_size = img.shape[0]
            
        x = self.conv_layers(img)
        
        score = self.score_layer(x.view(batch_size, -1))

        
        class_score = self.classifier(x.view(batch_size, -1))

        return score.view(batch_size), class_score