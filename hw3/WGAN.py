import torch
import numpy as np
from tqdm import tqdm_notebook as tqdm
import torch.utils.data 
from Generator import MyGenerator
from Discriminator import MyDiscriminator
import os
import matplotlib.pyplot as plt
class WGAN():
    def __init__(self, width=96, height=96, channels=3, clip_value=0.01, latent_dim = 1024,D_update_times = 5):

        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (self.width, self.height, self.channels)
        
        self.clip_value = clip_value
        self.latent_dim = latent_dim
        self.D_update_times = D_update_times
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.G = MyGenerator(self.shape, latent_dim).to(self.device)
        self.D = MyDiscriminator(self.shape).to(self.device)

        self.optimizer_G = torch.optim.RMSprop(self.G.parameters(), lr=2e-4)
        self.optimizer_D = torch.optim.RMSprop(self.D.parameters(), lr=2e-4)
        

        
    def train_D(self, X_train):
        ## train discriminator
        
        for p in self.D.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
            
        self.optimizer_D.zero_grad()

        
        real_images = X_train.float().to(self.device)
        #real_images = torch.tensor(real_images_tp, dtype=torch.float32, device = self.device)
        
        
        gen_noise = torch.tensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim)), dtype=torch.float32, device = self.device) 
        #with torch.no_grad():
        fake_images = self.G(gen_noise).detach()

        
        batch_d_loss = -torch.mean(self.D(real_images)) + torch.mean(self.D(fake_images))
        
        batch_d_loss.backward();
        self.optimizer_D.step();
        
        # Clip critic weights
        for p in self.D.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)
            
        return batch_d_loss
        
    def train_G(self):
        # train generator
        # gaussian noise
        noise = torch.tensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim)), dtype=torch.float32, device = self.device)  

        self.optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = self.G(noise)
#         image = gen_imgs[0, :, :, :].cpu().detach().numpy()
#         image = np.reshape(image, [self.height, self.width, self.channels])
#         plt.imshow(image)
#         plt.show()
#         input("")
        
        for p in self.D.parameters():
            p.requires_grad = False  # to avoid computation
            
        batch_g_loss = -torch.mean(self.D(gen_imgs))
        
        batch_g_loss.backward()
        self.optimizer_G.step()

        return batch_g_loss
        
    def _run_epoch(self, dataloader, training):
        self.D.train(training)
        self.G.train(training)
        
        if training:
            iter_in_epoch = min(len(dataloader), 1000000)
            description = 'train'
        else:
            iter_in_epoch = len(dataloader)
            description = 'test'
        grad_accumulate_steps = 1
        trange = tqdm(enumerate(dataloader), total=iter_in_epoch, desc=description)
        loss = 0
        
        for i, batch in trange:

            if training and i >= iter_in_epoch:
                break

            if training:
                d_loss = 0;
                #plt.imshow(batch[0])
                #plt.show()
                for d_ct in range(self.D_update_times):
                    d_loss += self.train_D(batch);
                d_loss /= self.D_update_times;    
                
                g_loss = self.train_G();
            else:
                with torch.no_grad():
                    print("predict not yet implement")


        d_loss = d_loss/iter_in_epoch
        g_loss = g_loss/iter_in_epoch
        
        return d_loss, g_loss

    
    def train(self, datas, epochs=10000, batch_size = 256, save_interval = 10):
        
        self.batch_size = batch_size
        
        for cnt in range(epochs):
            dataloader = torch.utils.data.DataLoader(datas, batch_size = batch_size, shuffle = True)
            
            d_loss, g_loss = self._run_epoch(dataloader, True)
            
            print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss, g_loss))

            if cnt % save_interval == 0:
                self.plot_images(save2file=False, step=cnt)
           
    def plot_images(self, save2file=False, samples=16, step=0):
        ''' Plot and generated images '''
        if not os.path.exists("./images"):
            os.makedirs("./images")
        filename = "./images/animate_%d.png" % step
        noise = torch.tensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim)), dtype=torch.float32, device = self.device) 
        #with torch.no_grad():
        images = self.G(noise)
#         image = images[0, :, :, :].cpu().detach().numpy()
#         image = np.reshape(image, [self.height, self.width, self.channels])
#         plt.imshow(image)
#         plt.show()
#         input("")
        
        plt.figure(figsize=(self.width, self.height))
        
        #for i in range(images.shape[0]):
        for i in range(16):
            plt.subplot(4, int(16/4), i+1)
            image = images[i, :, :, :].cpu().detach().numpy()
            image = np.reshape(image, [self.height, self.width, self.channels])
            #print(image)
            plt.imshow(image)
            #input("");
            #plt.axis('off')
            #plt.show()
        plt.tight_layout()

        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            print("plot figure:")
            plt.show()
