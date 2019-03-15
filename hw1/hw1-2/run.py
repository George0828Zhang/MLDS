import os

for model in range(3):
    for i in range(8):
        os.system("python model{}.py {}".format(model,i))