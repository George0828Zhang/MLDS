import torch
import torch.nn as nn
import numpy as np
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class MyGenerator(torch.nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(MyGenerator, self).__init__()
        #self.hidden_dim = 128
        self.img_shape = img_shape;      
        
        self.relu = torch.nn.LeakyReLU()
        self.dropout= torch.nn.Dropout(p=0.2, inplace=False)

       
        #self.Upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.linear = torch.nn.Linear(latent_dim, 128*16*16)
        self.deconv = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.d_BN=torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.conv2d_1 = torch.nn.Conv2d(128, 128, 4, padding =1 )
        self.BN1=torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.conv2d_2 = torch.nn.Conv2d(128, 64, 4, padding = 1)
        self.BN2=torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.conv2d_3 = torch.nn.Conv2d(64, 3, 4, padding = 3)
        self.BN3=torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        #print(x.shape)
        x = self.linear(x)
        
        x = self.relu(x).view(x.shape[0],128,16,16)
        x = self.dropout(x)
        #x = self.Upsample(x)
        
        x = self.deconv(x)
        x = self.d_BN(x)
        #print(x.shape)
        
        x = self.conv2d_1(x)
        x = self.BN1(x)
        
        x = self.relu(x)
        x = self.dropout(x)
        #print(x.shape)
        #x = self.Upsample(x)
        
        x = self.deconv(x)
        x = self.d_BN(x)
        #print(x.shape)
        
        x = self.conv2d_2(x)
        x = self.BN2(x);
        #print(x.shape)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2d_3(x)
        x = self.BN3(x)
        
        x = self.tanh(x)


#         print(x.shape)
#         input("")
        return x.transpose(2,1).transpose(3,2)
