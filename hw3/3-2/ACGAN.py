import torch
import numpy as np
from tqdm import tqdm_notebook as tqdm
import torch.utils.data 
from ACGAN_Generator import MyGenerator
from ACGAN_Discriminator import MyDiscriminator
import os
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import random
#from tensorboardX import SummaryWriter


class ACGAN():
    def __init__(self, width=64, height=64, channels=3, clip_value=0.01, latent_dim = 300, text_dim = 130, D_update_times = 1):

        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (self.width, self.height, self.channels)
        
        self.clip_value = clip_value
        self.latent_dim = latent_dim
        self.D_update_times = D_update_times
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.G = MyGenerator(self.shape, latent_dim, text_dim).to(self.device)
        self.D = MyDiscriminator(self.shape, text_dim).to(self.device)

        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr = 0.0003)
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr = 0.0002)
        #self.optimizer_D = torch.optim.RMSprop(self.D.parameters(), lr = 0.0002
        
        #fixed noise to generate 25 images for checking
        self.fixed_noise = torch.tensor(np.random.normal(0, 1, (5, self.latent_dim)), dtype=torch.float32, device = self.device)
        
        self.loss_BCE = torch.nn.BCELoss()
        self.loss_CE = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
        
        self.epoch = 0;
        
        #self.writer = SummaryWriter()
       
    def load(self, path):
        saved_model = torch.load(path)
        self.G.load_state_dict(saved_model['GModel'])
        self.D.load_state_dict(saved_model['DModel'])
        self.G = self.G.to(self.device)
        self.D = self.D.to(self.device)
        self.epoch = saved_model['epoch']

    
    def train_D(self, batch):#, miss_labels):
        ## train discriminator
        #print(len(tag))
        
        for p in self.D.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in train_G update
        
        for p in self.G.parameters():  # reset requires_grad
            p.requires_grad = False  
            
        
        real_images = batch['image']
        tag = batch['tag']        

        
        gen_noise = torch.tensor(np.random.normal(0, 1, (tag.shape[0], self.latent_dim)), dtype=torch.float32, device = self.device)
        fake_images = self.G(gen_noise.detach(), tag)
        



        self.optimizer_D.zero_grad()
        
        
        #score can be from -inf ~ inf, discriminator has no sigmoid, but loss_F has sigmoid
        real_score, tag_real = self.D(real_images)
        fake_score, tag_fake = self.D(fake_images)
        #wrong_label_real_score, _ = self.D(miss_labels, tag)
        miss_tag = np.zeros((tag.shape[0], tag.shape[1]))
        for i,t in enumerate(tag):            
            chosen = -1
            while(1):
                index = random.randint(0, tag.shape[1]-1)
                for ct,choose in enumerate(t):
                    if(choose == 1):
                        chosen = ct                       
                        break;
                if(chosen!=index):
                    miss_tag[i][index] = 1;
                    break;
        
        #miss_tag = torch.tensor(miss_tag).float().to(self.device)
        
        #wrong_label_real_score, _ = self.D(real_images)

        d_real_loss = self.loss_BCE(real_score, torch.ones(real_score.shape[0]).to(self.device))
        
        d_fake_loss = self.loss_BCE(fake_score, torch.zeros(fake_score.shape[0]).to(self.device))
        
        #d_miss_loss = self.loss_BCE(wrong_label_real_score, torch.zeros(fake_score.shape[0]).to(self.device))
        
        tag_real_loss = self.loss_CE(tag_real, tag.argmax(-1))
           
        tag_fake_loss = self.loss_CE(tag_fake, tag.argmax(-1))
        
        batch_d_loss = d_fake_loss + d_real_loss  + tag_real_loss + tag_fake_loss#+ d_miss_loss
        
        batch_d_loss.backward()
        
        self.optimizer_D.step();
       
        
        return batch_d_loss, real_score, fake_score
        
    def train_G(self, tag):
        # train generator
        # gaussian noise
        for p in self.G.parameters():  # reset requires_grad
            p.requires_grad = True 
            
        for p in self.D.parameters():
            p.requires_grad = False  # to avoid computation
            
        noise = torch.tensor(np.random.normal(0, 1, (tag.shape[0], self.latent_dim)), dtype=torch.float32, device = self.device) 

        self.optimizer_G.zero_grad()

        # Generate a batch of images
        gen_images = self.G(noise, tag)
#         image = gen_images[0, :, :, :].cpu().detach().numpy()
#         image = np.reshape(image, [self.height, self.width, self.channels])
#         plt.imshow(image)
#         plt.show()
#         input("")
        
        fake_score, tag_fake = self.D(gen_images)
            
        #g_fake_loss = self.loss_BCE(fake_score, torch.ones(tag.shape[0]).to(self.device));
        
        tag_fake_loss = self.loss_CE(tag_fake, tag.argmax(-1))
        
        batch_g_loss = -fake_score.mean() + tag_fake_loss
        
        batch_g_loss.backward()
        self.optimizer_G.step()

        return batch_g_loss
        
    def _run_epoch(self, datas, D_iters, training = True):
        self.D.train(training)
        self.G.train(training)
        
        dataloader = torch.utils.data.DataLoader(datas, batch_size = self.batch_size, shuffle = True, collate_fn = self.my_collate_fn)
        
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

                    #random_batch = sample(datas, len(batch['tag']))
                    #andom_img_batch = torch.tensor([data['image'] for data in random_batch]).float().to(self.device)

                    d_loss,d_real,d_fake = self.train_D(batch)#, random_img_batch);
                    d_loss_sum += d_loss.item()
                    d_real_sum += d_real.mean().item()
                    d_fake_sum += d_fake.mean().item()
                d_loss_sum /= D_iters; 
                d_real_sum /= D_iters;   
                d_fake_sum /= D_iters;   
                
                g_loss_sum += self.train_G(batch['tag']).item();
            else:
                with torch.no_grad():
                    
                    print("predict not yet implement")
  
            trange.set_postfix(
                **{'d_loss': '{:.3f}'.format(d_loss_sum/(i+1))},
                **{'g_loss': '{:.3f}'.format(g_loss_sum/(i+1))},
                **{'d_real': '{:.3f}'.format(d_real_sum/(i+1))},
                **{'d_fake ': '{:.3f}'.format(d_fake_sum/(i+1))}
                
            )


        d_loss_sum /= iter_in_epoch
        g_loss_sum /= iter_in_epoch
        d_real_sum /= iter_in_epoch
        d_fake_sum /= iter_in_epoch
        return d_loss_sum, g_loss_sum, d_real_sum, d_fake_sum

    def my_collate_fn(self, datas):
        batch = {}
        
        batch['image'] = torch.tensor([data['image'] for data in datas]).float().to(self.device)
        batch['tag'] = torch.tensor([data['tag'] for data in datas]).float().to(self.device)
        return batch
    
    def train(self, datas, epochs=10000, batch_size = 256, save_interval = 1):
        
        self.batch_size = batch_size
        
        dir_name = "./ACGANmodels"

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
         
        
        while(self.epoch < epochs):
            
            
            D_iters = self.D_update_times;
            
            d_loss, g_loss, d_real_loss, d_fake_loss = self._run_epoch(datas, D_iters, True)
            print ('epoch: %d, [Discriminator :: d_loss: %f], [Generator :: loss: %f], [d_fake :: score: %f], [d_real :: score: %f]' % (self.epoch, d_loss, g_loss, d_fake_loss, d_real_loss))
            

            if self.epoch % save_interval == 0:
                torch.save({
                'epoch': self.epoch,
                'GModel': self.G.state_dict(),
                'DModel': self.D.state_dict(),
                'd_loss': d_loss,
                'g_loss': g_loss,
                'd_real': d_real_loss,
                'd_fake': d_fake_loss,
                }, dir_name+"/ACGAN" + str(self.epoch))
                self.plot_images(save2file=True, step=self.epoch)
           
            self.epoch += 1
    
    def plot_images(self, save2file=True, step=0):
        ''' Plot and generated images '''
        if not os.path.exists("./ACGANimages"):
            os.makedirs("./ACGANimages")
        filename = "./ACGANimages/animate_%d.png" % step
        """
        #'black hair purple eyes'    : 89,
        #'pink hair black eyes'      : 94,
        #'red hair red eyes'         : 121,
        #'aqua hair green eyes'      : 100,
        #'blonde hair orange eyes'   : 2
        """
        tags = np.zeros((5,130))
        tags[0][89] = 1
        tags[1][94] = 1
        tags[2][121] = 1
        tags[3][100] = 1
        tags[4][2] = 1
        tags_tensor = torch.tensor(tags).float().to(self.device)
        #with torch.no_grad():
        images = self.G(self.fixed_noise, tags_tensor)
        #images is in -1 ~ 1
        rows = 1
        #turn -1 ~ 1 to 0 ~ 255
        image = (1 + images.cpu().detach().numpy())/2 * 255
        #image = np.reshape(image, [self.height, self.width, self.channels])
        int_X = image.clip(0,255).astype('uint8')

        int_X = int_X.reshape(rows, -1, self.height, self.width, self.channels)
        #print(int_X.shape)
        int_X = int_X.swapaxes(1,2).reshape(rows*self.height,-1, self.channels)
        image = Image.fromarray(int_X)


        a, classify_prob = self.D(images)
        b = torch.ones(tags_tensor.shape[0]).to(self.device)
        print(a)

        batch_g_loss = self.loss_BCE(a, b);
        classify_loss = self.loss_CE(classify_prob, tags_tensor.argmax(-1));
        print("plot g loss",batch_g_loss)
        print("classify loss", classify_loss)

        

            
        
        if save2file:
            image.save(filename)
            print("plot figure:")           
            display(image)
        else:
            print("plot figure:")           
            display(image)
            #print(image.shape)

