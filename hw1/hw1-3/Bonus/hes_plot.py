import numpy as np
import matplotlib.pyplot as plt

TRAIN_LOSS = np.load("TRAIN_LOSS.npy")
TEST_LOSS = np.load("TEST_LOSS.npy")
TRAIN_ACC = np.load("TRAIN_ACC.npy")
TEST_ACC = np.load("TEST_ACC.npy") 
HES_NORM = np.load("HES_NORM.npy" )


epsilon = 1e-4
sharpness = []
for i in range(len(HES_NORM)):
    sharp = HES_NORM[i] * (epsilon)**2 / (2*(1+TEST_LOSS[i])) 
    sharpness.append(sharp)
    
bs = [4,8,16,32,64,128,256,512,1024,2048][::-1]



plt.subplot(3,1,1)
plt.plot(bs, TRAIN_LOSS, 'b')
plt.plot(bs, TEST_LOSS, 'b:')
plt.xlabel('batch size(log scale)')
plt.xscale('log')
label = plt.ylabel('loss')
label.set_color("blue")
plt.legend(['train_loss','test_loss'], loc='upper left')


plt.twinx()


plt.plot(bs, sharpness, 'r')
plt.legend(['sharpness'], loc='upper right')
label = plt.ylabel('sharpness')
label.set_color("red")

plt.title("loss, sharpness v.s. batch_size")




#################################################3
plt.subplot(3,1,3)

plt.plot(bs, TRAIN_ACC, 'b')
plt.plot(bs, TEST_ACC, 'b:')
plt.xlabel('batch size(log scale)')
plt.xscale('log')
label = plt.ylabel('accuracy')
label.set_color("blue")
plt.legend(['train_acc','test_acc'], loc='upper left')


plt.twinx()


plt.plot(bs, sharpness, 'r')
plt.legend(['sharpness'], loc='upper right')
label = plt.ylabel('sharpness')
label.set_color("red")

plt.title("accuracy, sharpness v.s. batch_size")




#plt.show()
plt.savefig("sharpness vs batchsize.png")