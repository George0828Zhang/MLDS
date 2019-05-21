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
        self.emb_dim = 256
        
        self.embed = nn.Sequential(
            nn.Linear(text_dim, self.emb_dim),
            nn.ReLU(True)
        )
        
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
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (self.hidden_dim*8) x 4 x 4
            
            #nn.Conv2d(self.hidden_dim * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )
        
        #since using BCELogitloss there is no need sigmoid
        self.mlp_part2 = nn.Sequential(
            # state size. (self.hidden_dim*8) x 4 x 4
            nn.Conv2d(self.hidden_dim * 8 + self.emb_dim, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )
        
    def forward(self, img, txt):
        #print(img.shape)
        #print(txt.shape)
        
        if(img.shape[1] != 3):
            img = img.transpose(3,2).transpose(2,1)
        
        batch_size = img.shape[0]

        
        emb = self.embed(txt)
            
        #expect (batch, 256)
        #print("[debug][dis]",emb.shape)      
        
        #print("dis")

            
        out = self.mlp_part1(img)
        
        emb = emb.view(batch_size, self.emb_dim, 1, 1)
        emb = emb.repeat(1,1,4,4)
        #print("[debug][dis]",emb.shape)
        
        #print("[debug][dis]",out.shape)
       
        out = torch.cat((out, emb), 1)
        
        out = self.mlp_part2(out)

        return out.view(batch_size)  