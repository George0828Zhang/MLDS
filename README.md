# MLDS Lab
### HW 1-1
##### toolkits/libraries used and their corresponding version
- tensorflow 1.12
- tensorflow.keras
- sklearn 0.20.2 (PCA)

##### state how to reproduce the result shown in your presentation
1. To create model .h5 files and loss .npy files, run
```
python models.py
```
2. To plot the function image and loss image, run
```
python3 /hw1-1/plot.py
```
3. To show the loss, accuracy, weights, gradient norm plots of CIFAR10, open 
```
jupyter notebook /hw1/hw1-1/CIFAR10/CIFAR10_plot.ipynb
```
(However, the weights to produce weight plot is not available due to large filesize.)

4. To create the models of CIFAR10, open
```
jupyter notebook /hw1/hw1-1/CIFAR10/CIFAR10_DNN1.ipynb
jupyter notebook /hw1/hw1-1/CIFAR10/CIFAR10_DNN2.ipynb
jupyter notebook /hw1/hw1-1/CIFAR10/CIFAR10_SHALLOW1.ipynb
```

### HW 1-2
##### toolkits/libraries used and their corresponding version
Same as 1-1.

##### state how to reproduce the result shown in your presentation

1. To create 3 models training on the designed function
```
python run.py
```
2. To plot the images including PCA, loss, gradients
```
python designed_plot.py
```

3. To show minimal ratio results, open
```
jupyter notebook /hw1/hw1-2/mini_ratio_experiment.ipynb
```

### HW1-3
####
- tensorflow 1.12
- tensorflow.keras
1. To create the plot and the models of showing MNIST can fit random labels, open 
```
jupyter notebook /hw1/hw1-3/MNIST_RANDOM.ipynb
```
2. To create the plot and the models of showing parameters amount versus model performance, open
```
jupyter notebook /hw1/hw1-3/MNIST_GENERALIZATION.ipynb
```
3. To create train and plot for the task "flatness vs generalization part 1"
```
jupyter notebook /hw1/hw1-3/flatness/Flatness_pt1.ipynb
```
4.To create train and plot for the task "flatness vs generalization part 2"
```
jupyter notebook /hw1/hw1-3/flatness/Flatness_pt2.ipynb
```
5. To create the the requierments of showing sharpness versus batch size
```
python /hw1/hw1-3/bonus/hes.py
```
6. To create the plot of showing sharpness versus batch size
```
python /hw1/hw1-3/bonus/hes_plot.py
```

### HW2-1
#### toolkits
- tensorflow 1.12
- tensorflow.keras
1. To create dictionary
```
python make_dictionary.py
```
2. To train the basic seq2seq model, run
```
jupyter notebook /hw2/hw2-1/Training.ipynb
```
3. To train the best model
```
jupyter notebook /hw2/hw2-1/Training_v2.ipynb
```

### HW2-2
### toolkits
- pytorch 1.0.1
1. Run the WordEmbedding.ipynb
```
jupyter notebook /hw2/hw2-2/WordEmbedding.ipynb
```
Generate:
vocab.json
wv_matrix100d.npy
2. Train unidirectional model, run
```
jupyter notebook /hw2/hw2-2/pytorch_version.ipynb
```
3. Infer unidirectional model, run
```
jupyter notebook /hw2/hw2-2/BatchInference.ipynb
```
4. Train bidirectional model, run
```
jupyter notebook /hw2/hw2-2/Bidirectional.ipynb
```
5. Infer bidirectional model, run
```
jupyter notebook /hw2/hw2-2/BiAttBatchInference.ipynb
```
Trained model:
https://drive.google.com/file/d/1F5qJIRapmbtjSwDPvA8NY0ua_xVFxvQM/view?usp=sharing
