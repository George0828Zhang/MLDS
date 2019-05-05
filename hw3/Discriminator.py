import torch
import numpy as np
class MyDiscriminator(torch.nn.Module):
    def __init__(self, img_shape):
        super(MyDiscriminator, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(int(np.prod(img_shape)), 1024),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, 256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity  