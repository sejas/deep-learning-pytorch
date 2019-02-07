
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
batch = images.view(64,784)

### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 3 random normal variables
features = batch

print("features",features.shape)

# Define the size of each layer in our network
n_input = features.shape[1]     # 3 # Number of input units, must match number of input features
n_hidden = 256                    # Number of hidden units 
n_output = 10                    # Number of output units

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

## Your solution here
h = activation(torch.mm(features, W1) + B1) # 1,3 * 3,2 + 1,2 the result is (1, 2)
out = activation(torch.mm(h, W2) + B2) # 1,2 * 2,1 + 1,1 the result is (1, 1)
#print("h",h)
print("output",out)
print("output",out.shape)

#out = # output of your network, should have shape (64,10)
```

    features torch.Size([64, 784])
    output tensor([[1.1847e-08, 1.4722e-07, 7.8452e-02, 2.0188e-05, 9.9937e-01, 9.9990e-01,
             9.6378e-01, 9.9982e-01, 8.6483e-08, 9.8225e-01],
            [6.2230e-09, 4.4371e-07, 3.8055e-02, 4.2092e-05, 9.9960e-01, 1.0000e+00,
             3.0379e-07, 9.4261e-02, 3.6154e-08, 9.9754e-01],
            [4.5282e-04, 9.6967e-01, 8.9659e-01, 1.0719e-03, 9.9908e-01, 6.9857e-02,
             9.9295e-02, 1.6896e-01, 7.3137e-08, 2.8755e-02],
            [3.2858e-06, 4.4148e-02, 9.8756e-01, 1.9081e-03, 9.7738e-01, 9.9672e-01,
             9.9959e-01, 2.0253e-03, 2.5186e-09, 3.0712e-01],
            [3.0075e-04, 8.8240e-07, 1.4759e-01, 6.7459e-05, 4.3215e-01, 9.9994e-01,
             2.5706e-06, 1.5512e-02, 4.8345e-08, 4.6671e-03],
            [2.8054e-05, 1.6679e-01, 9.7072e-01, 1.0129e-01, 1.0000e+00, 9.9950e-01,
             1.8212e-04, 1.0000e+00, 2.7889e-02, 1.1216e-02],
            [1.5014e-05, 6.9144e-02, 4.6134e-03, 4.6539e-03, 2.2173e-01, 1.0000e+00,
             2.4212e-03, 9.4911e-01, 1.0173e-06, 4.9246e-04],
            [3.8225e-05, 1.3190e-01, 3.2483e-03, 1.7200e-09, 9.3786e-01, 9.9998e-01,
             2.5575e-02, 9.7567e-01, 1.3107e-06, 5.0557e-01],
            [1.4080e-05, 9.8199e-01, 7.9585e-01, 2.5346e-01, 9.7108e-01, 9.7877e-01,
             3.2826e-03, 3.0214e-02, 9.4403e-08, 1.6121e-02],
            [4.6218e-06, 5.3038e-03, 2.7545e-01, 2.7259e-07, 9.9998e-01, 1.0000e+00,
             1.1501e-06, 1.0312e-02, 2.1542e-05, 4.1730e-01],
            [2.0865e-04, 1.5702e-02, 9.6223e-02, 2.2247e-01, 9.9998e-01, 9.9991e-01,
             1.0276e-01, 9.9980e-01, 2.7259e-08, 9.9200e-01],
            [1.8745e-06, 7.5236e-06, 5.4968e-02, 5.4142e-04, 9.9984e-01, 9.9905e-01,
             8.9234e-05, 3.2432e-02, 5.8343e-04, 1.0469e-02],
            [9.6956e-08, 4.4157e-03, 1.0571e-01, 5.4162e-07, 9.8352e-02, 9.4912e-01,
             2.5089e-07, 5.9070e-01, 5.9867e-04, 9.7460e-01],
            [2.7438e-08, 1.4030e-01, 5.0043e-02, 6.0244e-05, 9.9984e-01, 9.9982e-01,
             6.2530e-05, 9.9979e-01, 1.2986e-04, 7.5902e-02],
            [1.0948e-07, 1.1024e-05, 2.3723e-03, 4.7273e-04, 9.4943e-01, 9.9998e-01,
             4.6054e-05, 2.1157e-03, 2.9345e-05, 6.2340e-05],
            [8.8620e-09, 3.4822e-04, 5.8622e-02, 3.7457e-03, 9.9985e-01, 9.9982e-01,
             3.6775e-02, 2.6888e-01, 1.1293e-03, 1.7823e-01],
            [1.5127e-04, 2.1872e-04, 9.2138e-01, 2.6856e-06, 9.9970e-01, 9.7490e-01,
             1.9511e-05, 1.3853e-02, 2.8844e-07, 3.7641e-01],
            [4.7862e-08, 2.1343e-03, 7.8739e-01, 6.2878e-02, 9.9172e-01, 9.9997e-01,
             4.4146e-01, 5.6927e-01, 4.3474e-09, 4.1890e-04],
            [2.6880e-04, 2.3303e-05, 9.8923e-01, 2.2714e-07, 9.9998e-01, 9.9998e-01,
             1.0535e-03, 9.9999e-01, 5.4432e-08, 3.3169e-02],
            [1.3929e-09, 8.9261e-07, 3.5139e-01, 6.4713e-07, 9.9956e-01, 9.9950e-01,
             3.3017e-05, 9.9842e-01, 2.0413e-05, 6.9016e-01],
            [3.4724e-04, 9.3801e-04, 1.9962e-02, 6.8584e-07, 8.3125e-01, 1.0000e+00,
             2.6328e-06, 3.9070e-02, 1.3628e-04, 1.2988e-01],
            [3.2698e-08, 1.0077e-04, 5.4462e-01, 6.8001e-08, 9.9962e-01, 9.9984e-01,
             5.4466e-03, 9.1791e-01, 4.4195e-08, 9.9399e-01],
            [4.0023e-04, 2.0416e-03, 8.8625e-01, 5.4785e-09, 9.6460e-01, 9.9995e-01,
             1.6321e-04, 9.9992e-01, 1.2190e-06, 4.5001e-03],
            [2.2583e-08, 1.4406e-03, 9.8919e-01, 1.7629e-07, 9.9963e-01, 9.8784e-01,
             8.1424e-01, 9.9955e-01, 6.0872e-08, 3.8881e-01],
            [6.5591e-08, 1.1564e-03, 8.1954e-02, 2.7912e-06, 9.9989e-01, 9.7747e-01,
             9.7889e-03, 9.8651e-01, 1.5060e-03, 4.7200e-06],
            [9.0677e-07, 7.1560e-03, 7.0212e-01, 3.4986e-01, 9.9970e-01, 8.0776e-01,
             1.5533e-06, 2.1825e-05, 3.5550e-05, 9.9554e-01],
            [4.5855e-07, 2.5599e-05, 3.4606e-01, 1.9624e-09, 9.9999e-01, 9.9292e-01,
             2.3448e-04, 9.5648e-01, 5.1206e-06, 2.2933e-02],
            [5.1360e-03, 3.1154e-05, 9.3909e-01, 7.1387e-05, 9.9980e-01, 7.4222e-01,
             1.3544e-04, 3.5014e-01, 1.1593e-02, 5.5307e-03],
            [1.5912e-04, 2.7341e-03, 9.9998e-01, 3.2423e-03, 1.0000e+00, 9.9687e-01,
             1.3846e-05, 9.9812e-01, 6.8755e-06, 4.2306e-01],
            [1.3129e-04, 1.9085e-03, 9.9980e-01, 4.7847e-03, 9.9995e-01, 9.9999e-01,
             3.0130e-02, 9.5036e-01, 1.1179e-06, 5.4783e-08],
            [1.0539e-07, 2.0544e-07, 1.1504e-05, 3.9090e-04, 9.9991e-01, 9.9486e-01,
             1.5214e-01, 9.9939e-01, 3.4395e-06, 6.4736e-03],
            [2.1466e-05, 1.8051e-02, 8.0662e-02, 1.8205e-07, 9.9375e-01, 9.9906e-01,
             5.5694e-01, 9.9982e-01, 4.8639e-09, 9.8977e-01],
            [3.4273e-09, 2.6156e-02, 8.4526e-03, 6.0646e-10, 5.6708e-02, 9.9988e-01,
             1.0550e-03, 9.9652e-01, 2.3775e-07, 1.2921e-02],
            [1.8343e-07, 1.4288e-07, 4.4336e-01, 6.2212e-07, 6.6333e-01, 9.9985e-01,
             2.7167e-05, 9.9900e-01, 1.4591e-05, 6.7560e-01],
            [4.3827e-05, 1.4402e-07, 9.9604e-01, 1.1728e-04, 9.8282e-01, 8.8260e-01,
             1.0891e-03, 9.4155e-01, 3.0136e-05, 4.5564e-04],
            [3.7878e-07, 3.7731e-04, 6.8168e-03, 9.3348e-07, 9.9174e-01, 9.8927e-01,
             2.6936e-04, 9.9996e-01, 2.2636e-03, 2.0923e-02],
            [2.8405e-05, 3.7500e-05, 9.8754e-01, 1.3853e-05, 9.9997e-01, 9.9468e-01,
             2.8812e-04, 1.7049e-05, 2.0954e-04, 2.9906e-06],
            [2.3349e-01, 5.6016e-01, 9.5778e-01, 4.7021e-05, 9.9993e-01, 1.0000e+00,
             1.5078e-03, 9.8733e-01, 1.2169e-08, 9.9995e-01],
            [1.8030e-05, 1.7849e-07, 3.6550e-01, 8.9711e-01, 9.9988e-01, 9.9997e-01,
             2.0375e-01, 3.6745e-02, 4.5807e-04, 5.3336e-06],
            [4.2603e-10, 1.4231e-03, 3.7384e-01, 3.7555e-07, 9.9999e-01, 9.8913e-01,
             2.9132e-03, 9.9809e-01, 4.3747e-05, 9.4297e-01],
            [4.3060e-10, 7.9033e-01, 8.1361e-01, 2.2487e-06, 9.9901e-01, 1.6660e-01,
             9.5341e-06, 9.9911e-01, 3.2418e-06, 1.5481e-03],
            [4.5367e-04, 1.1061e-05, 9.7759e-01, 6.1124e-07, 9.9740e-01, 9.9999e-01,
             9.9805e-01, 5.7214e-01, 1.6728e-08, 9.3314e-04],
            [5.9743e-11, 2.6942e-01, 9.0872e-01, 9.9986e-01, 9.9854e-01, 1.0000e+00,
             2.5191e-02, 1.4059e-02, 1.9604e-05, 1.2625e-04],
            [2.4979e-09, 9.6664e-04, 1.0000e+00, 1.1411e-04, 9.9908e-01, 9.9785e-01,
             3.5526e-05, 9.9996e-01, 2.9881e-07, 4.8494e-01],
            [3.9619e-07, 1.4422e-05, 7.4751e-02, 2.5593e-01, 9.9953e-01, 9.9809e-01,
             6.3580e-01, 6.7990e-01, 1.9490e-06, 5.4324e-02],
            [2.6191e-07, 5.8688e-07, 1.5810e-03, 6.1544e-02, 1.0000e+00, 1.7976e-01,
             9.8373e-05, 5.2161e-04, 4.1840e-07, 5.0375e-04],
            [1.6238e-05, 1.3057e-05, 2.7570e-01, 5.0761e-04, 9.9969e-01, 9.9972e-01,
             5.7657e-01, 9.8869e-01, 5.7273e-07, 1.5491e-02],
            [9.7258e-10, 1.6900e-04, 7.5753e-02, 8.0079e-06, 9.6123e-01, 3.0004e-02,
             2.9539e-07, 9.0423e-03, 3.0190e-06, 2.9441e-02],
            [1.1761e-06, 6.9112e-04, 8.6552e-01, 1.8075e-03, 9.8301e-01, 5.5687e-01,
             6.7563e-05, 4.0616e-01, 2.3613e-06, 4.1608e-05],
            [1.1663e-09, 6.7943e-06, 9.9999e-01, 3.5685e-03, 9.9657e-01, 1.0000e+00,
             5.6263e-03, 9.9982e-01, 2.3971e-05, 5.9081e-03],
            [1.9983e-10, 7.1171e-06, 5.2402e-01, 5.7955e-01, 9.9627e-01, 9.9908e-01,
             6.6287e-09, 9.9407e-01, 4.2514e-07, 8.5707e-01],
            [2.2785e-06, 6.0078e-03, 8.7497e-01, 8.6739e-01, 9.9989e-01, 9.9923e-01,
             4.0800e-03, 6.2934e-03, 7.4631e-07, 9.9161e-01],
            [2.2719e-07, 1.9874e-02, 6.4732e-01, 3.1613e-03, 9.9342e-01, 9.9794e-01,
             2.5075e-02, 2.8618e-01, 1.0907e-07, 7.3502e-01],
            [3.1824e-08, 4.0505e-01, 7.2291e-01, 4.1612e-06, 8.3243e-01, 1.0000e+00,
             1.8761e-03, 1.9782e-04, 1.0004e-09, 1.6990e-03],
            [1.2049e-04, 9.9879e-05, 9.9838e-01, 1.4254e-05, 9.9971e-01, 9.9117e-01,
             4.3809e-07, 7.8295e-02, 3.7435e-07, 3.1816e-02],
            [5.4716e-10, 1.2537e-05, 7.6989e-02, 3.8787e-04, 9.9996e-01, 9.9375e-01,
             4.6823e-02, 9.9660e-01, 7.2874e-06, 7.5534e-03],
            [1.0562e-04, 2.3214e-04, 1.5837e-04, 4.7945e-10, 4.7324e-02, 9.9968e-01,
             3.3591e-07, 7.2088e-01, 6.8414e-10, 5.0011e-02],
            [2.4772e-07, 1.2273e-04, 1.6401e-04, 3.4755e-09, 9.7313e-01, 9.9965e-01,
             9.9799e-01, 6.8679e-01, 6.3146e-07, 8.5824e-01],
            [1.4666e-05, 9.5759e-01, 7.7182e-01, 2.1535e-05, 1.0000e+00, 9.9902e-01,
             1.0712e-03, 8.7059e-01, 8.8163e-06, 2.8712e-01],
            [1.3864e-05, 7.8197e-05, 5.6366e-04, 2.1784e-06, 4.9080e-01, 9.9996e-01,
             7.3764e-06, 1.7715e-02, 1.1729e-07, 3.3500e-01],
            [2.2941e-09, 2.4643e-05, 3.1987e-02, 3.3444e-10, 9.9990e-01, 9.9997e-01,
             8.6306e-03, 9.7048e-01, 1.5892e-04, 1.6016e-02],
            [1.9700e-05, 3.3619e-01, 9.1536e-01, 1.2361e-06, 9.9961e-01, 8.3345e-01,
             1.7987e-01, 6.2403e-05, 6.2512e-05, 9.5094e-01],
            [1.0302e-03, 1.7999e-08, 1.1001e-02, 1.6251e-01, 9.9973e-01, 1.0000e+00,
             1.5309e-02, 9.9977e-01, 5.6104e-06, 3.3458e-01],
            [9.9555e-06, 1.3012e-04, 2.7632e-01, 2.4760e-04, 9.9923e-01, 9.9955e-01,
             1.2500e-02, 4.2955e-01, 4.6558e-06, 8.1051e-01]])
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
    ## TODO: Implement the softmax function here

# Here, out should be the output of the network in the previous excercise with shape (64,10)
probabilities = softmax(out)

# Does it have the right shape? Should be (64, 10)
print(probabilities.shape)
# Does it sum to 1?
print(probabilities.sum(dim=1))
```

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

```

### Initializing weights and biases

The weights and such are automatically initialized for you, but it's possible to customize how they are initialized. The weights and biases are tensors attached to the layer you defined, you can get them with `model.fc1.weight` for instance.


```python
print(model.fc1.weight)
print(model.fc1.bias)
```

For custom initialization, we want to modify these tensors in place. These are actually autograd *Variables*, so we need to get back the actual tensors with `model.fc1.weight.data`. Once we have the tensors, we can fill them with zeros (for biases) or random normal values.


```python
# Set biases to all zeros
model.fc1.bias.data.fill_(0)
```


```python
# sample from random normal with standard dev = 0.01
model.fc1.weight.data.normal_(std=0.01)
```

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

Here our model is the same as before: 784 input units, a hidden layer with 128 units, ReLU activation, 64 unit hidden layer, another ReLU, then the output layer with 10 units, and the softmax output.

The operations are available by passing in the appropriate index. For example, if you want to get first Linear operation and look at the weights, you'd use `model[0]`.


```python
print(model[0])
model[0].weight
```

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

Now you can access layers either by integer or the name


```python
print(model[0])
print(model.fc1)
```

In the next notebook, we'll see how we can train a neural network to accuractly predict the numbers appearing in the MNIST images.
