import torch
import numpy as np
from tqdm import tqdm_notebook as tqdm
import torch.utils.data 
from Generator import MyGenerator
from Discriminator import MyDiscriminator
import os
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display

class WGAN():
    def __init__(self, width=64, height=64, channels=3, clip_value=0.01, latent_dim = 300,D_update_times = 10):

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

        self.optimizer_G = torch.optim.RMSprop(self.G.parameters(), lr=1e-4)
        self.optimizer_D = torch.optim.RMSprop(self.D.parameters(), lr=1e-4)
        
        self.epoch = 0;
       
    def load(self, path):
        saved_model = torch.load(path)
        self.G.load_state_dict(saved_model['GModel'])
        self.D.load_state_dict(saved_model['DModel'])
        self.G = self.G.to(self.device)
        self.D = self.D.to(self.device)
        self.epoch = saved_model['epoch']

    def train_D(self, X_train):
        ## train discriminator
        
        for p in self.D.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in train_G update
        
        for p in self.G.parameters():  # reset requires_grad
            p.requires_grad = False  
        
        self.optimizer_D.zero_grad()

        
        
        real_images = X_train.float().to(self.device)
        #real_images = torch.tensor(real_images_tp, dtype=torch.float32, device = self.device)
#         print(X_train)
#         print(real_images)
        
        gen_noise = torch.tensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim)), dtype=torch.float32, device = self.device) 
        #with torch.no_grad():
        fake_images = self.G(gen_noise)

#         #debug
#         image = fake_images[0, :, :, :].cpu().detach().numpy()
#         print(image)
#         image = (1 + image)/2 * 255
#         image = image.clip(0,255).astype('uint8')
#         #print(image.shape);
#         #image = np.reshape(image, [self.height, self.width, self.channels])
#         #print(image.shape)
#         #print(image)
#         plt.imshow(image)
#         plt.show()
        
#         image = real_images[0, :, :, :].cpu().detach().numpy()
#         print(image)
#         image = (1 + image)/2 * 255
#         image = image.clip(0,255).astype('uint8')
#         image = np.reshape(image, [self.height, self.width, self.channels])
#         #print(image)
#         plt.imshow(image)
#         plt.show()
#         input("")
        
        
        real_loss = torch.mean(self.D(real_images)) 
        fake_loss = torch.mean(self.D(fake_images))
        #print(real_loss)
        #print(fake_loss)
        #input("")
        batch_d_loss = -real_loss + fake_loss
        batch_d_loss.backward();
        
        # Clip critic weights
        for p in self.D.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)
        
        self.optimizer_D.step();
        return batch_d_loss, real_loss, fake_loss
        
    def train_G(self):
        # train generator
        # gaussian noise
        for p in self.G.parameters():  # reset requires_grad
            p.requires_grad = True 
            
        for p in self.D.parameters():
            p.requires_grad = False  # to avoid computation
            
        noise = torch.tensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim)), dtype=torch.float32, device = self.device)  

        self.optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = self.G(noise)
#         image = gen_imgs[0, :, :, :].cpu().detach().numpy()
#         image = np.reshape(image, [self.height, self.width, self.channels])
#         plt.imshow(image)
#         plt.show()
#         input("")
        

            
        batch_g_loss = -torch.mean(self.D(gen_imgs))
        
        batch_g_loss.backward()
        self.optimizer_G.step()

        return batch_g_loss
        
    def _run_epoch(self, dataloader, D_iters, training = True):
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
        g_loss_sum = 0
        d_loss_sum = 0;
        d_real_sum = 0;
        d_fake_sum = 0;
        
        for i, batch in trange:            
            
            if training and i >= iter_in_epoch:
                break

            if training:
                
                #plt.imshow(batch[0])
                #plt.show()
                for d_ct in range(D_iters):
                    d_loss,d_real,d_fake = self.train_D(batch);
                    d_loss_sum += d_loss
                    d_real_sum += d_real
                    d_fake_sum += d_fake
                d_loss_sum /= D_iters; 
                d_real_sum /= D_iters;   
                d_fake_sum /= D_iters;   
                
                g_loss_sum += self.train_G();
            else:
                with torch.no_grad():
                    print("predict not yet implement")


        d_loss_sum /= iter_in_epoch
        g_loss_sum /= iter_in_epoch
        d_real_sum /= iter_in_epoch
        d_fake_sum /= iter_in_epoch
        return d_loss_sum, g_loss_sum, d_real_sum, d_fake_sum

    
    def train(self, datas, epochs=10000, batch_size = 256, save_interval = 10):
        
        self.batch_size = batch_size
        #fixed noise to generate 64 images for checking
        self.fixed_noise = torch.tensor(np.random.normal(0, 1, (64, self.latent_dim)), dtype=torch.float32, device = self.device) 
        
        while(self.epoch < epochs):
            dataloader = torch.utils.data.DataLoader(datas, batch_size = batch_size, shuffle = True)
            
            if( self.epoch < 5 or self.epoch % save_interval == 0):
                D_iters =  10 * self.D_update_times;
            else:
                D_iters = self.D_update_times;
            
            d_loss, g_loss, d_real_loss, d_fake_loss = self._run_epoch(dataloader, D_iters, True)
            print ('epoch: %d, [Discriminator :: d_loss: %f], [Generator :: loss: %f], [d_fake :: loss: %f], [d_real :: loss: %f]' % (self.epoch, d_loss, g_loss, d_fake_loss, d_real_loss))

            if self.epoch % save_interval == 0:
                self.plot_images(d_loss, g_loss, d_fake_loss, d_real_loss, save2file=False, step=self.epoch)
           
            self.epoch += 1
    
    def plot_images(self, d_loss, g_loss, d_fake_loss, d_real_loss, save2file=False, samples=16, step=0):
        ''' Plot and generated images '''
        if not os.path.exists("./images"):
            os.makedirs("./images")
        filename = "./images/animate_%d.png" % step

        #with torch.no_grad():
        images = self.G(self.fixed_noise)
        #images is in -1 ~ 1
        rows = 8
        #turn -1 ~ 1 to 0 ~ 255
        image = (1 + images.cpu().detach().numpy())/2 * 255
        #image = np.reshape(image, [self.height, self.width, self.channels])
        int_X = image.clip(0,255).astype('uint8')
        int_X = int_X.reshape(rows, -1, self.height, self.width, self.channels)
        #print(int_X.shape)
        int_X = int_X.swapaxes(1,2).reshape(rows*self.height,-1, self.channels)
        img = Image.fromarray(int_X)
        
        if not os.path.exists("./models"):
            os.makedirs("./models")
        torch.save({
        'epoch': step,
        'GModel': self.G.state_dict(),
        'DModel': self.D.state_dict(),
        'd_loss': d_loss,
        'g_loss': g_loss,
        'd_real': d_real_loss,
        'd_fake': d_fake_loss,
        }, "./models/WGAN" + str(step))
        
        if save2file:
            img.save(filename)
        else:
            print("plot figure:")           
            display(img)
            #print(img.shape)

