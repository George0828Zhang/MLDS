import torch
import numpy as np
class MyDiscriminator(torch.nn.Module):
    def __init__(self, img_shape):
        super(MyDiscriminator, self).__init__()
        self.hidden_dim = 32
        #int(np.prod(img_shape))
        self.model = torch.nn.Sequential(            
            torch.nn.Conv2d(3, self.hidden_dim, 5, stride=2, padding=2),   
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, 5, stride=2, padding=2),  
            torch.nn.LeakyReLU(True),
            torch.nn.Conv2d(2 * self.hidden_dim, 4 * self.hidden_dim, 5, stride=2, padding=2),
            torch.nn.LeakyReLU(True),
        )
        self.linear = torch.nn.Linear(4 * self.hidden_dim * 8 * 8, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, img):
        #print("dis")
        if(img.shape[1] != 3):
            img = img.transpose(3,2).transpose(2,1)
        out = self.model(img)
        #print(out.shape) 
        out = out.view(img.shape[0], -1)
        #print(img.shape[0])
        #print(out.shape)
        out = self.linear(out)
        out = self.sigmoid(out)
        #print(out.shape) 
        return out.squeeze(-1)  