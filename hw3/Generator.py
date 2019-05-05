import torch
import numpy as np
class MyGenerator(torch.nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(MyGenerator, self).__init__()
        
        self.img_shape = img_shape;
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, int(np.prod(self.img_shape))),
            torch.nn.Tanh()
        )
    def forward(self, z):
        img = self.model(z)
        img = img.view(z.shape[0], self.img_shape[0], self.img_shape[1], self.img_shape[2])
        return img
#     def __init__(self):
#         super(Generator, self).__init__()


#         self.linear = torch.nn.Linear(100, 128*16*16)
#         self.relu = torch.nn.ReLU()
#         self.Upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')  
#         self.conv2d_1 = torch.nn.Conv2d(128, 128, 4)
#         self.conv2d_2 = torch.nn.Conv2d(256, 64, 4)
#         self.conv2d_3 = torch.nn.Conv2d(128, 3, 4)
#         self.tanh = torch.nn.Tanh()
#     def forward(self, x):
#         print(x.shape)
#         x = self.linear(x)
#         print(x.shape)
#         x = self.relu(x).view(16,16,128)
#         x = self.Upsample(x)
#         x = self.conv2d_1(x)
#         x = self.relu(x)
#         x = self.Upsample(x)
#         x = self.conv2d_2(x)
#         x = self.tanh(x)
#         return x
