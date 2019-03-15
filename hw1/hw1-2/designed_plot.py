import matplotlib.pyplot as plt
from pylab import *
import numpy as np
epoch = 300
base_dir = "./designed_pca_loss_grads"

pca_x_w = [[],[],[]]
pca_y_w = [[],[],[]]
pca_x_f = [[],[],[]]
pca_y_f = [[],[],[]]

'''
pca_x = [[],[],[]]
pca_y = [[],[],[]]
'''
loss = [[],[],[]]  #shape(400,1)
grads = [[],[],[]]

###########################load data###########################################
for model in range(3):
    for event in range(8):        
        
        pca_x_w[model].append(np.load(base_dir + "/model{}_x_w_{}.npy".format(model,event)))
        pca_y_w[model].append(np.load(base_dir + "/model{}_y_w_{}.npy".format(model,event)))        
        pca_x_f[model].append(np.load(base_dir + "/model{}_x_f_{}.npy".format(model,event)))
        pca_y_f[model].append(np.load(base_dir + "/model{}_y_f_{}.npy".format(model,event)))
        
        
        '''
        pca_x[model].append(np.load(base_dir + "/model{}_x_{}.npy".format(model,event)))
        pca_y[model].append(np.load(base_dir + "/model{}_y_{}.npy".format(model,event)))
        '''
        loss[model].append(np.load(base_dir + "/model{}_loss_{}.npy".format(model,event)))
        grads[model].append(np.load(base_dir + "/model{}_grads_{}.npy".format(model,event)))
        
        

############################plot pca###########################################
colors = ["red", "gold","green","blue","black","brown","pink","purple"]
for model in range(3):
    for i in range(8):
        plt.scatter(pca_x_w[model][i], pca_y_w[model][i], c=colors[i])    
    plt.savefig("model{}_whole_PCA.png".format(model))  
plt.close()

for model in range(3):
    for i in range(8):
        plt.scatter(pca_x_f[model][i], pca_y_f[model][i], c=colors[i])    
    plt.savefig("model{}_first_PCA.png".format(model))  
plt.close()



############################plot loss##########################################

#a little preprocessing
for i in range(3):
    for j in range(8):
        loss[i][j] = loss[i][j].reshape((300,))
        
for model in range(3):
    for event in range(8):
        plt.subplot(2,1,1)
        plt.title("model{}_event{}".format(model,event))
        plt.ylabel("loss")
        plt.plot(range(epoch), loss[model][event])    
            
        
        
        ############plot grads################
        plt.subplot(2,1,2)
        plt.ylabel("gradient_norm")
        plt.xlabel("epoch")
        plt.plot(range(epoch), grads[model][event])            
        plt.tight_layout()
        
        plt.savefig("model{}_event{}".format(model,event))
        plt.close()
    
    
    
    
    
    
    
    
    
    