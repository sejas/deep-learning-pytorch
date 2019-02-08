
# Neural networks with PyTorch

Deep learning networks tend to be massive with dozens or hundreds of layers, that's where the term "deep" comes from. You can build one of these deep networks using only weight matrices as we did in the previous notebook, but in general it's very cumbersome and difficult to implement. PyTorch has a nice module `nn` that provides a nice way to efficiently build large neural networks.


```python
# Import necessary packages

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import numpy as np
import torch

import helper

import matplotlib.pyplot as plt
```


Now we're going to build a larger network that can solve a (formerly) difficult problem, identifying text in an image. Here we'll use the MNIST dataset which consists of greyscale handwritten digits. Each image is 28x28 pixels, you can see a sample below

<img src='assets/mnist.png'>

Our goal is to build a neural network that can take one of these images and predict the digit in the image.

First up, we need to get our dataset. This is provided through the `torchvision` package. The code below will download the MNIST dataset, then create training and test datasets for us. Don't worry too much about the details here, you'll learn more about this later.


```python
### Run this cell

from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Processing...
    Done!


We have the training data loaded into `trainloader` and we make that an iterator with `iter(trainloader)`. Later, we'll use this to loop through the dataset for training, like

```python
for image, label in trainloader:
    ## do things with images and labels
```

You'll notice I created the `trainloader` with a batch size of 64, and `shuffle=True`. The batch size is the number of images we get in one iteration from the data loader and pass through our network, often called a *batch*. And `shuffle=True` tells it to shuffle the dataset every time we start going through the data loader again. But here I'm just grabbing the first batch so we can check out the data. We can see below that `images` is just a tensor with size `(64, 1, 28, 28)`. So, 64 images per batch, 1 color channel, and 28x28 images.


```python
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)
```

    <class 'torch.Tensor'>
    torch.Size([64, 1, 28, 28])
    torch.Size([64])
    tensor(3)


This is what one of the images looks like. 


```python
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');
print(labels[1])
```

    tensor(9)



![png](output_7_1.png)


First, let's try to build a simple network for this dataset using weight matrices and matrix multiplications. Then, we'll see how to do it using PyTorch's `nn` module which provides a much more convenient and powerful method for defining network architectures.

The networks you've seen so far are called *fully-connected* or *dense* networks. Each unit in one layer is connected to each unit in the next layer. In fully-connected networks, the input to each layer must be a one-dimensional vector (which can be stacked into a 2D tensor as a batch of multiple examples). However, our images are 28x28 2D tensors, so we need to convert them into 1D vectors. Thinking about sizes, we need to convert the batch of images with shape `(64, 1, 28, 28)` to a have a shape of `(64, 784)`, 784 is 28 times 28. This is typically called *flattening*, we flattened the 2D images into 1D vectors.

Previously you built a network with one output unit. Here we need 10 output units, one for each digit. We want our network to predict the digit shown in an image, so what we'll do is calculate probabilities that the image is of any one digit or class. This ends up being a discrete probability distribution over the classes (digits) that tells us the most likely class for the image. That means we need 10 output units for the 10 classes (digits). We'll see how to convert the network output into a probability distribution next.

> **Exercise:** Flatten the batch of images `images`. Then build a multi-layer network with 784 input units, 256 hidden units, and 10 output units using random tensors for the weights and biases. For now, use a sigmoid activation for the hidden layer. Leave the output layer without an activation, we'll add one that gives us a probability distribution next.


```python
def activation(x):
    """ Sigmoid activation function 
    
        Arguments
        ---------
        x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))
```


```python
## Your solution
# Flattering
batch = images.view(64,-1) # images.view(64,784)

### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 3 random normal variables
features = batch

print("features",features.shape)

# Define the size of each layer in our network
n_input = features.shape[1]     # Number of input units, must match number of input features
n_hidden = 256                  # Number of hidden units 
n_output = 10                   # Number of output units

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

## Your solution here
h = activation(torch.mm(features, W1) + B1) # 64,784 * 784,256 + 1,256 the result is (64, 256)
out = torch.mm(h, W2) + B2 # 64,256 * 256,10 + 1,10 the result is (64, 10)
#print("h",h)
print("output",out)
print("output",out.shape)

#out = # output of your network, should have shape (64,10)
```

    features torch.Size([64, 784])
    output tensor([[-18.2512, -15.7314,  -2.4636, -10.8104,   7.3659,   9.2355,   3.2812,
               8.6170, -16.2633,   4.0136],
            [-18.8950, -14.6281,  -3.2299, -10.0756,   7.8268,  12.9792, -15.0069,
              -2.2627, -17.1355,   6.0042],
            [ -7.6996,   3.4649,   2.1599,  -6.8372,   6.9916,  -2.5889,  -2.2051,
              -1.5930, -16.4309,  -3.5198],
            [-12.6259,  -3.0751,   4.3742,  -6.2597,   3.7662,   5.7175,   7.8055,
              -6.2000, -19.7995,  -0.8136],
            [ -8.1089, -13.9406,  -1.7536,  -9.6039,  -0.2731,   9.8022, -12.8714,
              -4.1505, -16.8449,  -5.3625],
            [-10.4814,  -1.6085,   3.5010,  -2.1830,  13.4998,   7.5912,  -8.6107,
              17.3793,  -3.5512,  -4.4791],
            [-11.1065,  -2.5999,  -5.3742,  -5.3654,  -1.2556,  14.6052,  -6.0211,
               2.9258, -13.7983,  -7.6156],
            [-10.1720,  -1.8843,  -5.7264, -20.1810,   2.7141,  11.0542,  -3.6402,
               3.6915, -13.5450,   0.0223],
            [-11.1707,   3.9985,   1.3606,  -1.0803,   3.5139,   3.8308,  -5.7158,
              -3.4688, -16.1757,  -4.1114],
            [-12.2847,  -5.2340,  -0.9671, -15.1153,  11.0128,  13.5432, -13.6757,
              -4.5641, -10.7455,  -0.3338],
            [ -8.4746,  -4.1382,  -2.2399,  -1.2513,  10.7537,   9.2732,  -2.1669,
               8.5121, -17.4179,   4.8209],
            [-13.1872, -11.7975,  -2.8445,  -7.5208,   8.7572,   6.9594,  -9.3242,
              -3.3956,  -7.4460,  -4.5489],
            [-16.1490,  -5.4182,  -2.1353, -14.4287,  -2.2157,   2.9261, -15.1982,
               0.3669,  -7.4202,   3.6471],
            [-17.4113,  -1.8128,  -2.9435,  -9.7170,   8.7652,   8.6223,  -9.6798,
               8.4669,  -8.9490,  -2.4994],
            [-16.0275, -11.4154,  -6.0415,  -7.6565,   2.9325,  11.0964,  -9.9857,
              -6.1563, -10.4363,  -9.6828],
            [-18.5415,  -7.9623,  -2.7762,  -5.5834,   8.7785,   8.6021,  -3.2655,
              -1.0003,  -6.7850,  -1.5284],
            [ -8.7963,  -8.4275,   2.4613, -12.8276,   8.1080,   3.6597, -10.8445,
              -4.2653, -15.0588,  -0.5048],
            [-16.8549,  -6.1475,   1.3092,  -2.7016,   4.7852,  10.4470,  -0.2352,
               0.2789, -19.2537,  -7.7775],
            [ -8.2213, -10.6669,   4.5205, -15.2977,  11.0904,  10.8281,  -6.8546,
              12.0137, -16.7263,  -3.3724],
            [-20.3919, -13.9291,  -0.6129, -14.2507,   7.7206,   7.5992, -10.3184,
               6.4484, -10.7993,   0.8009],
            [ -7.9651,  -6.9708,  -3.8938, -14.1926,   1.5945,  14.3521, -12.8475,
              -3.2025,  -8.9006,  -1.9020],
            [-17.2359,  -9.2026,   0.1789, -16.5037,   7.8627,   8.7297,  -5.2073,
               2.4143, -16.9347,   5.1081],
            [ -7.8231,  -6.1920,   2.0530, -19.0224,   3.3050,   9.9607,  -8.7203,
               9.4196, -13.6174,  -5.3991],
            [-17.6061,  -6.5412,   4.5167, -15.5511,   7.8974,   4.3974,   1.4778,
               7.7024, -16.6145,  -0.4523],
            [-16.5398,  -6.7613,  -2.4161, -12.7890,   9.1165,   3.7701,  -4.6167,
               4.2926,  -6.4968, -12.2637],
            [-13.9134,  -4.9326,   0.8574,  -0.6197,   8.1179,   1.4355, -13.3751,
             -10.7324, -10.2445,   5.4086],
            [-14.5952, -10.5729,  -0.6364, -20.0491,  11.8670,   4.9430,  -8.3579,
               3.0900, -12.1822,  -3.7520],
            [ -5.2663, -10.3765,   2.7354,  -9.5473,   8.4936,   1.0576,  -8.9069,
              -0.6184,  -4.4457,  -5.1919],
            [ -8.7457,  -5.8992,  11.0624,  -5.7282,  13.6373,   5.7636, -11.1875,
               6.2732, -11.8875,  -0.3102],
            [ -8.9380,  -6.2595,   8.5293,  -5.3375,   9.9978,  11.4070,  -3.4716,
               2.9519, -13.7041, -16.7199],
            [-16.0656, -15.3981, -11.3728,  -7.8467,   9.2701,   5.2659,  -1.7179,
               7.4075, -12.5802,  -5.0335],
            [-10.7490,  -3.9963,  -2.4334, -15.5190,   5.0696,   6.9717,   0.2288,
               8.6291, -19.1414,   4.5720],
            [-19.4915,  -3.6172,  -4.7648, -21.2234,  -2.8115,   9.0108,  -6.8532,
               5.6571, -15.2520,  -4.3359],
            [-15.5114, -15.7613,  -0.2275, -14.2901,   0.6782,   8.8181, -10.5135,
               6.9102, -11.1351,   0.7336],
            [-10.0352, -15.7533,   5.5282,  -9.0509,   4.0469,   2.0172,  -6.8213,
               2.7793, -10.4098,  -7.6934],
            [-14.7863,  -7.8821,  -4.9815, -13.8843,   4.7882,   4.5240,  -8.2192,
              10.1498,  -6.0886,  -3.8458],
            [-10.4689, -10.1911,   4.3725, -11.1870,  10.2719,   5.2311,  -8.1519,
             -10.9794,  -8.4704, -12.7200],
            [ -1.1887,   0.2418,   3.1218,  -9.9649,   9.5446,  17.6311,  -6.4956,
               4.3554, -18.2244,   9.8407],
            [-10.9235, -15.5388,  -0.5516,   2.1655,   9.0116,  10.4240,  -1.3630,
              -3.2663,  -7.6880, -12.1415],
            [-21.5765,  -6.5535,  -0.5158, -14.7949,  11.3367,   4.5107,  -5.8356,
               6.2587, -10.0370,   2.8054],
            [-21.5658,   1.3269,   1.4736, -13.0051,   6.9144,  -1.6099, -11.5606,
               7.0277, -12.6394,  -6.4692],
            [ -7.6977, -11.4121,   3.7755, -14.3078,   5.9509,  11.2467,   6.2372,
               0.2906, -17.9062,  -6.9760],
            [-23.5410,  -0.9976,   2.2981,   8.9015,   6.5282,  14.7645,  -3.6558,
              -4.2504, -10.8398,  -8.9771],
            [-19.8078,  -6.9407,  15.5594,  -9.0783,   6.9854,   6.1409, -10.2452,
              10.1427, -15.0235,  -0.0603],
            [-14.7414, -11.1467,  -2.5159,  -1.0672,   7.6590,   6.2584,   0.5572,
               0.7533, -13.1482,  -2.8569],
            [-15.1552, -14.3485,  -6.4481,  -2.7245,  13.5835,  -1.5180,  -9.2266,
              -7.5581, -14.6868,  -7.5929],
            [-11.0281, -11.2462,  -0.9659,  -7.5853,   8.0758,   8.1847,   0.3087,
               4.4705, -14.3728,  -4.1519],
            [-20.7511,  -8.6854,  -2.5015, -11.7351,   3.2107,  -3.4760, -15.0350,
              -4.6968, -12.7106,  -3.4955],
            [-13.6533,  -7.2765,   1.8619,  -6.3140,   4.0579,   0.2285,  -9.6024,
              -0.3799, -12.9563, -10.0872],
            [-20.5694, -11.8994,  11.4142,  -5.6320,   5.6705,  12.2225,  -5.1747,
               8.6199, -10.6387,  -5.1255],
            [-22.3336, -11.8530,   0.0962,   0.3209,   5.5884,   6.9868, -18.8319,
               5.1210, -14.6708,   1.7911],
            [-12.9920,  -5.1087,   1.9456,   1.8781,   9.1432,   7.1672,  -5.4976,
              -5.0619, -14.1081,   4.7722],
            [-15.2975,  -3.8983,   0.6073,  -5.7536,   5.0170,   6.1842,  -3.6605,
              -0.9140, -16.0313,   1.0203],
            [-17.2630,  -0.3845,   0.9589, -12.3897,   1.6029,  14.8238,  -6.2767,
              -8.5280, -20.7229,  -6.3760],
            [ -9.0238,  -9.2114,   6.4241, -11.1585,   8.1586,   4.7202, -14.6408,
              -2.4657, -14.7981,  -3.4154],
            [-21.3263, -11.2868,  -2.4840,  -7.8545,  10.1933,   5.0688,  -3.0134,
               5.6795, -11.8294,  -4.8782],
            [ -9.1556,  -8.3679,  -8.7504, -21.4584,  -3.0023,   8.0475, -14.9064,
               0.9488, -21.1029,  -2.9442],
            [-15.2110,  -9.0054,  -8.7154, -19.4775,   3.5895,   7.9474,   6.2089,
               0.7852, -14.2752,   1.8008],
            [-11.1300,   3.1170,   1.2186, -10.7458,  17.5263,   6.9255,  -6.8379,
               1.9062, -11.6389,  -0.9094],
            [-11.1862,  -9.4562,  -7.4805, -13.0369,  -0.0368,  10.0133, -11.8172,
              -4.0155, -15.9586,  -0.6857],
            [-19.8929, -10.6110,  -3.4099, -21.8186,   9.1674,  10.4589,  -4.7438,
               3.4929,  -8.7469,  -4.1180],
            [-10.8349,  -0.6803,   2.3810, -13.6036,   7.8460,   1.6103,  -1.5172,
              -9.6818,  -9.6801,   2.9645],
            [ -6.8769, -17.8329,  -4.4987,  -1.6397,   8.2182,  13.6534,  -4.1639,
               8.3806, -12.0909,  -0.6875],
            [-11.5174,  -8.9469,  -0.9628,  -8.3035,   7.1650,   7.7034,  -4.3694,
              -0.2837, -12.2774,   1.4533]])
    output torch.Size([64, 10])


Now we have 10 outputs for our network. We want to pass in an image to our network and get out a probability distribution over the classes that tells us the likely class(es) the image belongs to. Something that looks like this:
<img src='assets/image_distribution.png' width=500px>

Here we see that the probability for each class is roughly the same. This is representing an untrained network, it hasn't seen any data yet so it just returns a uniform distribution with equal probabilities for each class.

To calculate this probability distribution, we often use the [**softmax** function](https://en.wikipedia.org/wiki/Softmax_function). Mathematically this looks like

$$
\Large \sigma(x_i) = \cfrac{e^{x_i}}{\sum_k^K{e^{x_k}}}
$$

What this does is squish each input $x_i$ between 0 and 1 and normalizes the values to give you a proper probability distribution where the probabilites sum up to one.

> **Exercise:** Implement a function `softmax` that performs the softmax calculation and returns probability distributions for each example in the batch. Note that you'll need to pay attention to the shapes when doing this. If you have a tensor `a` with shape `(64, 10)` and a tensor `b` with shape `(64,)`, doing `a/b` will give you an error because PyTorch will try to do the division across the columns (called broadcasting) but you'll get a size mismatch. The way to think about this is for each of the 64 examples, you only want to divide by one value, the sum in the denominator. So you need `b` to have a shape of `(64, 1)`. This way PyTorch will divide the 10 values in each row of `a` by the one value in each row of `b`. Pay attention to how you take the sum as well. You'll need to define the `dim` keyword in `torch.sum`. Setting `dim=0` takes the sum across the rows while `dim=1` takes the sum across the columns.


```python
def softmax(x):
    ## DONE: Implement the softmax function here
    #           64      /         (1, 64).view(64,1) (like transpose)
    # We want to divide 64 by just one value.
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)
# Here, out should be the output of the network in the previous excercise with shape (64,10)
probabilities = softmax(out)

# Does it have the right shape? Should be (64, 10)
print(probabilities.shape)
# Does it sum to 1?
print(probabilities.sum(dim=1))
```

    torch.Size([64, 10])
    tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000])


## Building networks with PyTorch

PyTorch provides a module `nn` that makes building networks much simpler. Here I'll show you how to build the same one as above with 784 inputs, 256 hidden units, 10 output units and a softmax output.


```python
from torch import nn
```


```python
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x
```

Let's go through this bit by bit.

```python
class Network(nn.Module):
```

Here we're inheriting from `nn.Module`. Combined with `super().__init__()` this creates a class that tracks the architecture and provides a lot of useful methods and attributes. It is mandatory to inherit from `nn.Module` when you're creating a class for your network. The name of the class itself can be anything.

```python
self.hidden = nn.Linear(784, 256)
```

This line creates a module for a linear transformation, $x\mathbf{W} + b$, with 784 inputs and 256 outputs and assigns it to `self.hidden`. The module automatically creates the weight and bias tensors which we'll use in the `forward` method. You can access the weight and bias tensors once the network (`net`) is created with `net.hidden.weight` and `net.hidden.bias`.

```python
self.output = nn.Linear(256, 10)
```

Similarly, this creates another linear transformation with 256 inputs and 10 outputs.

```python
self.sigmoid = nn.Sigmoid()
self.softmax = nn.Softmax(dim=1)
```

Here I defined operations for the sigmoid activation and softmax output. Setting `dim=1` in `nn.Softmax(dim=1)` calculates softmax across the columns.

```python
def forward(self, x):
```

PyTorch networks created with `nn.Module` must have a `forward` method defined. It takes in a tensor `x` and passes it through the operations you defined in the `__init__` method.

```python
x = self.hidden(x)
x = self.sigmoid(x)
x = self.output(x)
x = self.softmax(x)
```

Here the input tensor `x` is passed through each operation and reassigned to `x`. We can see that the input tensor goes through the hidden layer, then a sigmoid function, then the output layer, and finally the softmax function. It doesn't matter what you name the variables here, as long as the inputs and outputs of the operations match the network architecture you want to build. The order in which you define things in the `__init__` method doesn't matter, but you'll need to sequence the operations correctly in the `forward` method.

Now we can create a `Network` object.


```python
# Create the network and look at it's text representation
model = Network()
model
```




    Network(
      (hidden): Linear(in_features=784, out_features=256, bias=True)
      (output): Linear(in_features=256, out_features=10, bias=True)
      (sigmoid): Sigmoid()
      (softmax): Softmax()
    )



You can define the network somewhat more concisely and clearly using the `torch.nn.functional` module. This is the most common way you'll see networks defined as many operations are simple element-wise functions. We normally import this module as `F`, `import torch.nn.functional as F`.


```python
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)
        
        return x
```

### Activation functions

So far we've only been looking at the sigmoid activation function, but in general any function can be used as an activation function. The only requirement is that for a network to approximate a non-linear function, the activation functions must be non-linear. Here are a few more examples of common activation functions: Tanh (hyperbolic tangent), and ReLU (rectified linear unit).

<img src="assets/activation.png" width=700px>

In practice, the ReLU function is used almost exclusively as the activation function for hidden layers.

### Your Turn to Build a Network

<img src="assets/mlp_mnist.png" width=600px>

> **Exercise:** Create a network with 784 input units, a hidden layer with 128 units and a ReLU activation, then a hidden layer with 64 units and a ReLU activation, and finally an output layer with a softmax activation as shown above. You can use a ReLU activation with the `nn.ReLU` module or `F.relu` function.

It's good practice to name your layers by their type of network, for instance 'fc' to represent a fully-connected layer. As you code your solution, use `fc1`, `fc2`, and `fc3` as your layer names.


```python
## Your solution here
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden1 layer linear transformation
        self.fc1 = nn.Linear(784, 128)
        # hidden1 to hidden2 layer linear transformation
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer with softmax activation
        x = F.softmax(self.fc3(x), dim=1)
        
        return x
model = Network()
```

### Initializing weights and biases

The weights and such are automatically initialized for you, but it's possible to customize how they are initialized. The weights and biases are tensors attached to the layer you defined, you can get them with `model.fc1.weight` for instance.


```python
print(model.fc1.weight)
print(model.fc1.bias)
```

    Parameter containing:
    tensor([[-0.0299, -0.0127, -0.0251,  ...,  0.0306, -0.0006,  0.0221],
            [-0.0095, -0.0079, -0.0285,  ..., -0.0207, -0.0022, -0.0159],
            [-0.0288,  0.0111,  0.0197,  ...,  0.0109,  0.0006, -0.0051],
            ...,
            [ 0.0191,  0.0355, -0.0216,  ..., -0.0224, -0.0000, -0.0083],
            [-0.0290, -0.0005,  0.0196,  ..., -0.0142,  0.0130, -0.0232],
            [ 0.0165,  0.0093, -0.0166,  ...,  0.0234,  0.0222,  0.0203]],
           requires_grad=True)
    Parameter containing:
    tensor([ 0.0262,  0.0296, -0.0108, -0.0064,  0.0180, -0.0047,  0.0105,  0.0042,
             0.0050, -0.0133, -0.0083,  0.0133, -0.0317, -0.0146, -0.0031,  0.0171,
             0.0288,  0.0029,  0.0147,  0.0295, -0.0013,  0.0049,  0.0247, -0.0326,
            -0.0055, -0.0231, -0.0069, -0.0187, -0.0333,  0.0343, -0.0063, -0.0141,
             0.0327,  0.0091, -0.0101, -0.0188,  0.0301,  0.0256, -0.0108,  0.0093,
             0.0089,  0.0093, -0.0292,  0.0131,  0.0256, -0.0138,  0.0153,  0.0273,
             0.0066,  0.0345, -0.0128,  0.0190,  0.0034,  0.0117,  0.0134,  0.0144,
             0.0037, -0.0099, -0.0171, -0.0095,  0.0201,  0.0027, -0.0062,  0.0028,
             0.0265,  0.0031,  0.0345, -0.0004,  0.0325, -0.0343,  0.0099,  0.0169,
             0.0140, -0.0341,  0.0286, -0.0260, -0.0175,  0.0157,  0.0248, -0.0128,
             0.0161, -0.0168, -0.0167,  0.0019,  0.0188,  0.0110,  0.0313, -0.0031,
            -0.0108,  0.0241, -0.0161, -0.0078, -0.0036, -0.0139,  0.0260,  0.0125,
             0.0187,  0.0048, -0.0068, -0.0301, -0.0230,  0.0031, -0.0267, -0.0196,
            -0.0258,  0.0187, -0.0288,  0.0161, -0.0343,  0.0192, -0.0294, -0.0337,
             0.0057,  0.0257, -0.0113, -0.0332,  0.0268, -0.0126, -0.0142, -0.0063,
             0.0066,  0.0109, -0.0091,  0.0304,  0.0030,  0.0057,  0.0276,  0.0053],
           requires_grad=True)


For custom initialization, we want to modify these tensors in place. These are actually autograd *Variables*, so we need to get back the actual tensors with `model.fc1.weight.data`. Once we have the tensors, we can fill them with zeros (for biases) or random normal values.


```python
# Set biases to all zeros
model.fc1.bias.data.fill_(0)
```




    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0.])




```python
# sample from random normal with standard dev = 0.01
model.fc1.weight.data.normal_(std=0.01)
```




    tensor([[-0.0123,  0.0081, -0.0036,  ...,  0.0064, -0.0245,  0.0069],
            [-0.0020, -0.0137,  0.0014,  ..., -0.0133, -0.0197, -0.0017],
            [ 0.0023,  0.0064, -0.0153,  ..., -0.0032,  0.0068,  0.0049],
            ...,
            [-0.0204,  0.0093, -0.0026,  ...,  0.0080, -0.0147, -0.0107],
            [ 0.0171,  0.0051, -0.0079,  ..., -0.0056,  0.0075, -0.0063],
            [-0.0066, -0.0029,  0.0032,  ...,  0.0046,  0.0079,  0.0024]])



### Forward pass

Now that we have a network, let's see what happens when we pass in an image.


```python
# Grab some data 
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels) 
images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to automatically get batch size

# Forward pass through the network
img_idx = 0
ps = model.forward(images[img_idx,:])

img = images[img_idx]
helper.view_classify(img.view(1, 28, 28), ps)
```


![png](output_29_0.png)


As you can see above, our network has basically no idea what this digit is. It's because we haven't trained it yet, all the weights are random!

### Using `nn.Sequential`

PyTorch provides a convenient way to build networks like this where a tensor is passed sequentially through operations, `nn.Sequential` ([documentation](https://pytorch.org/docs/master/nn.html#torch.nn.Sequential)). Using this to build the equivalent network:


```python
# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)

# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])
helper.view_classify(images[0].view(1, 28, 28), ps)
```

    Sequential(
      (0): Linear(in_features=784, out_features=128, bias=True)
      (1): ReLU()
      (2): Linear(in_features=128, out_features=64, bias=True)
      (3): ReLU()
      (4): Linear(in_features=64, out_features=10, bias=True)
      (5): Softmax()
    )



![png](output_31_1.png)


Here our model is the same as before: 784 input units, a hidden layer with 128 units, ReLU activation, 64 unit hidden layer, another ReLU, then the output layer with 10 units, and the softmax output.

The operations are available by passing in the appropriate index. For example, if you want to get first Linear operation and look at the weights, you'd use `model[0]`.


```python
print(model[0])
model[0].weight
```

    Linear(in_features=784, out_features=128, bias=True)





    Parameter containing:
    tensor([[ 0.0058, -0.0231, -0.0009,  ...,  0.0188, -0.0200, -0.0254],
            [-0.0232,  0.0331, -0.0269,  ...,  0.0011, -0.0096,  0.0317],
            [ 0.0028,  0.0284, -0.0117,  ...,  0.0202,  0.0230,  0.0005],
            ...,
            [-0.0069,  0.0162,  0.0140,  ..., -0.0214,  0.0297,  0.0184],
            [ 0.0096,  0.0194, -0.0018,  ...,  0.0244,  0.0015, -0.0216],
            [ 0.0272, -0.0186, -0.0313,  ..., -0.0191, -0.0354, -0.0250]],
           requires_grad=True)



You can also pass in an `OrderedDict` to name the individual layers and operations, instead of using incremental integers. Note that dictionary keys must be unique, so _each operation must have a different name_.


```python
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(hidden_sizes[1], output_size)),
                      ('softmax', nn.Softmax(dim=1))]))
model
```




    Sequential(
      (fc1): Linear(in_features=784, out_features=128, bias=True)
      (relu1): ReLU()
      (fc2): Linear(in_features=128, out_features=64, bias=True)
      (relu2): ReLU()
      (output): Linear(in_features=64, out_features=10, bias=True)
      (softmax): Softmax()
    )



Now you can access layers either by integer or the name


```python
print(model[0])
print(model.fc1)
```

    Linear(in_features=784, out_features=128, bias=True)
    Linear(in_features=784, out_features=128, bias=True)


In the next notebook, we'll see how we can train a neural network to accuractly predict the numbers appearing in the MNIST images.
