import torch
import torch.nn as nn
import numpy as np
class MyDiscriminator(torch.nn.Module):
    def __init__(self, img_shape, text_dim):
        super(MyDiscriminator, self).__init__()
        self.hidden_dim = 32
        #int(np.prod(img_shape))
        self.n_channel = 3
        self.text_dim = text_dim
        
        self.mlp_part1 = nn.Sequential(
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

        
        self.toOne = nn.Conv2d(self.hidden_dim * 8, 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        self.linear = torch.nn.Linear(4096, text_dim)
        
#         self.Alex = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(2304, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, text_dim),
#         )
        
        
    def forward(self, img):
        #print(img.shape)
        #print(txt.shape)
        
        if(img.shape[1] != 3):
            img = img.transpose(3,2).transpose(2,1)
        
        batch_size = img.shape[0]
            
        #expect (batch, 256)
        #print("[debug][dis]",emb.shape)      
        
        #print("dis")

        #print(img.shape)    
        x = self.mlp_part1(img)
        #print(x.shape)
        y_1 = self.toOne(x)
        #out = self.linear(y_1.view(batch_size, -1))
        #out = self.sigmoid(out)
        #y_1 = self.sigmoid(y_1);
        
        y_2 = self.linear(x.view(batch_size, -1))
        #print(x)
        #print(out)
        #input("")
        return y_1.view(batch_size), y_2