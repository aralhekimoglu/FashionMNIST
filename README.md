# Fashion MNIST

Fashion MNIST is a dataset that consists of 60k images for 10 clothing categories. It has been collected with the purpose of replacing the MNIST challenge, which is now considered an easy challenge.

In this repository, some architectures, optimizers, augmentation methods and other hyper parameters are experimented. Convolutional neural networks is proven to perform great for vision tasks at feature extraction. Therefore, I choose to use CNN architectures as the main feature extractor and experimented with different number of CNNs to use, using a residual block, batch normalization and dropout, channel dimensions, size of convolution filters. 

There are numerous optimization methods and strategies to use when training CNNs so in this repo I experimented with different optimizers like SGD and Adam, I tried different learning rates and different learning rate schedulers. Even though Fashion MNIST is more challenging than MNIST, with today's neural network architectures, it is easy to overfit, therefore additional data augmentation strategies are tried like horizontal flipping the image, erasing some parts of the image and translating the image. 

All of these experiments are described in the following section and code, along with experiments results are shared.

# Usage

## Libraries

numpy
torch
torchvision
PIL
tqdm

## Dataset
Dataset can be downloaded using the torchvision library.

## Training/ Testing

train.py contains the code to train and test at the same time and write the experiment results to the folder given as argument. For more detail one can look at the train.sh script for arguments to use. Experiment folder has a structure as following:

- <b> ckpt:</b> Save model checkpoints during training.
- <b>losses.png:</b> Training loss versus epoch graph
- <b>accuracies.png: </b> Test accuracy versus epoch graph
- <b>arguments.txt:</b> Saves the arguments used for running that  experiment
- <b>train.log:</b> Keeps a log file of terminal during training process.

Experiment folders mentioned in the following section can be downloaded from <a href=" https://drive.google.com/open?id=1FG8arL3IyJIW-0_D0ZyuzAv7fUE1AOYI">here </a>

# Experiments

This is a classification task and accuracy is used to evaluate the model performance. For time complexity of the models, test time latencies are compared, which is the forward propogation time for a single batch image with no augmentations.

## Different Network Structures
In this experimental setting, base channel dimension is 32 and kernel size is 3, batch normalization is used, dropout is set to 0.2, initial learning rate is set to 0.01, batch size for training is 256 and Adam optimizer is used for training. Following results are average of multiple runs with the same settings.

| Network Structure  | # of Parameters   |Test Accuracy(%)      | Test Latency(ms) |
|--------------------|-------------------|----------------------| -----------------|
|2 Conv   		     | 50K               |92.84                 | 0.75 
|2 Conv + Residual Block | 76K           |93.11                 | 1.61
|4 Conv              | 247K              | <b>93.14             | 1.15
|4 Conv + Residual Block |402K           |92.98                 |2.84

Based on the results, using residual blocks is not necessary at a task like this where only small number of convolutional blocks are used, so learning do not suffer from lack of gradient flow and increasing the parameter count does not help the network.

Also test time latency also increases with more layers, therefore using fewer layers increases the speed.


## Convolutional Channel Dimension

This parameters is given to the network as input and is the output channel dimension of the first convolution. At each convolution step output channel dimension is doubled. 4 Convolution + Residual block setting from the first experiment are used here.


|  Channel Dimension           | # of Parameters   |Test Accuracy(%)      | Test Latency(ms) 
|----------------|---| --------------------------| -----------------------------|
|32   		 | 402 K |<b>92.98            | 2.84
|64 | 1.48 M          |92.48          | 2.83
|128          | 5.67 M | 92.70|  2.82
|512 |87.54 M          |91.46 | 2.83

Results show that using a higher channel dimension does not increase the test accuracy and cause more overfitting. So moving on with the simplest of these models which is 32 channel dimension makes more sense considering Occam's Razor. 

An interesting remark to make about parallel programming, looking at the test time latency of these different models are that they do not differ significantly based on the channel dimension. The reason is the parallel programming abilities of GPUs and since all these operations are done in parallel for each channel dimension, speed is not affected, although the number of operations increases significantly.

## Kernel Size

This is size of the convolutional kernels used to extract features for the next feature map.

|  Kernel Size           | # of Parameters   |Test Accuracy(%)      | 
|----------------|-----------------------------| -----------------------------|
|1   		 | 83 K |89.09            |
|3 | 247 K          |<b>93.14          |
|5          | 576 K | 92.92|

Since results between 3 and 5 are similar I will move on with the one with smaller number of parameters, i.e. simpler model, which is kernel size with 3, which is also faster at test time.


## Effect of Batch Normalization

|  Batchnorm           | # of Parameters   |Test Accuracy(%)      | 
|----------------|-----------------------------| -----------------------------|
|No   		 | 247 K |92.62            |
|Yes | 247 K          |<b>93.14          |

Batch normalization is useful for convolutions and prevent overfitting as well as faster convergence and based on the results, they are useful even at a toy task like this and have so few parameters that barely change total.

## Effect of dropout

Dropout is applied to the final feature vector learned before feeding it to a fully connected layer for classification.

|  Dropout Ratio           | # of Parameters   |Test Accuracy(%)      | 
|----------------|-----------------------------| -----------------------------|
|0   		 | 247 K |93.01            |
|0.2 | 247 K          |93.14          |
|0.5          | 247 K | <b>93.25|

Dropout offers a solution to overfitting no extra parameters. As the dropout ratio is increased, test accuracy increases.

## Optimizers

Optimizer choice can effect both the speed and the final value of convergence. Here I tested with SGD(with momentum at PyTorch) and Adam optimizer.

|  Optimizer           | Test Accuracy(%)      |  Training Loss
|--------------------|-------------------------| --- |
|Adam  		  |<b>93.14            | <b>4.42
|SGD        |92.65         | 6.18
  
### Learning Rate

Choice of learning rate is important for deep learning applications.


|  Learning Rate           | Test Accuracy(%)      |  Training Loss
|--------------------|-------------------------| -- |
|0.1  		  |91.83            | 48.04 
|0.05        |92.34         | 22.71 
|0.01        |93.14         | 4.42
|0.005        |<b>93.18         | <b>2.70

Based on the results, choosing a high learning rate disabled optimizer to converge to smaller training loss value. Although using a smaller learning rate would slow down the process of converging, with a small dataset like this, this problem does not occur.

## Learning Rate Scheduler

As shown in the previous section, learning rate significantly affects the optimizer's ability to converge to a smaller loss value. This can be dealt with picking a smaller learning rate, or even better, using a learning rate scheduler to decrease the learning rate as the optimizer gets closer to minima. Here 2 learning rate schedulers are tested. Step scheduler decreases the learning rate to 1/10 of its value every 10 steps. Exponential scheduler multiplies the learning rate with 0.99 every step.

|  LR Scheduler           | Test Accuracy(%)      |  Training Loss
|--------------------|-------------------------| -- |
|Step  		  |<b>93.14            | 4.42 
|Expo        |92.86         | <b>0.05 
 
 Using exponential learning rate, model converges to a smaller value for training loss however this is clearly overfitting and not ideal, because the test accuracy decreases. 
 
## Augmentations

Data augmentation techniques are really important at computer vision tasks, because with just some simple adjustments, it is easy to increase the size of your training data quite significantly without messing up with the ground truth labels. In this repository, I experimented with 3 common data augmentations and their combinations.

|  Augmentation Method(s)           | Test Accuracy(%)      |  
|--------------------|-------------------------| 
|None 		  |93.14            |
|Horizontal Flip  		  |93.47            |
|Random Erasing        |93.41      |
|Random Translation        |93.35      |
|HFlip and RErasing        |<b> 92.53      |
|HFlip and RTranslation        |92.92      |
|RErasing and RTranslation          |93.12      |
|All methods        |92.78      |

Each method separately improves the test accuracy, but combining all does not improve the performance. Using just Horizontal Flip and Random Erasing seems to be the best augmentation technique for this dataset.