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

3. To show the loss, accuracy, weights, gradient norm plots, open 
```
jupyter notebook /hw1-1/CIFAR10/CIFAR10_plot.ipynb
```
(However, the weights to produce weight plot is not available due to large filesize.)

4. To show minimal ratio results, open
```
jupyter notebook mini_ratio_experiment.ipynb
```
