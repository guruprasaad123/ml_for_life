# Building a neural network from scratch

Artificial Neural Networks are really inspired by the biological neural networks inside our brains..In which each neurons fires when when we act , move , decide to complete whatever we started. [Readmore...](https://en.wikipedia.org/wiki/Artificial_neural_network)

## Step 1: Building Basic Blocks ( Neurons )

When we are started to construct a neural network , we need to construct the basic blocks that forms the neural net that is *Neurons* , Basically a Neuron takes in some N-inputs , does some math and outputs-M.

![alt text](https://cdn-images-1.medium.com/max/800/1*JRRC_UDsW1kDgPK3MW1GjQ.png)

**The Neuron performs 3 actions here,**

First , the inputs (x) are multiplied by weights (w)

![alt text](https://cdn-images-1.medium.com/max/800/1*Iq76QGqSTJfYRztFhwK0yw.png)

Second, the result is added with a bias (b)

![alt text](https://cdn-images-1.medium.com/max/800/1*CE-YfWhFQ2yQSGq9Zaxd9Q.png)

Third , sum is passed through an activation function f(x)

![5c8fd2144018b](https://cdn-images-1.medium.com/max/800/1*9BFMXPkoAqN_EW7XTPvuGg.png)

[Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) takes the any Numeric input ( -infinity , +infinity ) and outputs in the range of 0 and 1.There are many activation functions out there like [tanh](https://en.wikipedia.org/wiki/Hyperbolic_function) , [relu](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#relu) , [adam](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) which optimies more than Sigmoid .But we are not to use here in this example.And you can more about activation functions [here](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html).

**And This is how we calculate the sigmoid function** , f(x) denotes sigmoid function

$$
f(x) = \frac{1}{1+e^{-x}} 

$$

Note : f(x) denotes sigmoid function , f'(x) denotes derivative of f(x) which we will use later.

$$
f'(x) = \frac{e^x}{(1+e^{-x})^2} = f(x) * (1-f(x))
$$

**Here's the graph of how sigmoid function works.**

![](https://cdn-images-1.medium.com/max/800/1*Ul8Yu_r8GKSFillzbPFrPQ.png)

## Step 2 : Connecting Neurons to form a Network

Neural Network = Bunch of neurons connected together to form a network.Here's how a simple neural network looks like.

![](https://cdn-images-1.medium.com/max/800/1*JuCFYUaqd7WTX8PKHkfuQw.png)

Things to notice :

**Input_layer** : The layers where the inputs are passed to a neural network in general . ( x1 , x2 )

**Hidden_layer** : This layer is where mainly the computation is going on.Numbers of hidden-layers , neurons depands upon the complexity of our task.Here's a guide to[ it.](https://www.heatonresearch.com/2017/06/01/hidden-layers.html). ( h1 , h2 )

**Output_layer** : This layers may contains one or more layers ( o1 )

## Step 3 : Picking a Loss function

For this example we'll be using **Mean-squared-loss** function .But there are many more loss functions out there you can also check that [out](https://en.wikipedia.org/wiki/Loss_function).

![](https://cdn-images-1.medium.com/max/800/1*8Fn15kWdz4VpPymonQuHGg.png)

Then , we will have derive a *multi-variable-loss-function* like this.

![](https://cdn-images-1.medium.com/max/800/1*OHMn7EMtIG77EAmccwgowg.png)

## Step 4 : Performing Stochastic-Gradient-descent , BackPropogation.

What we'll need to adjust the weights ( w1...w4 ) , bias ( b1...b3) so that our loss can be minimum.lets put it into a simple expression  (∂*L/∂*w*1*​) .

$$
\frac{change~of~L}{because~of~the~change~in~weights}

$$

which can be further expanded to this form .

![](https://cdn-images-1.medium.com/max/800/1*Ojh2mA6NWye18NnTUne24Q.png)

let's start calculating (∂*L*/∂*y_pred*​) .

![](https://cdn-images-1.medium.com/max/800/1*wKP2ce3tNUUj-vsKjQsSvw.png)

y_pred involves calculating the weights ( w5 , w6 ) and bias ( b3 ) , so 

![](https://cdn-images-1.medium.com/max/800/1*IAPqA69MXq8_fwQeYcv7bg.png)

In ,  (∂*y_pred*​/∂*w1*) w1 affects h1 alone so , our derivation is as follows .

![](https://cdn-images-1.medium.com/max/800/1*pFc6605O7Xf0lI8l7cWsEw.png)

We do the same thing for ( ∂*w*1/​∂*h*1​​ ).

![](https://cdn-images-1.medium.com/max/800/1*gBYqQUbNNB0pSGfR0pDq2w.png)

Breaking down (∂*L/∂*w*1*​) , we got this derivation :

![](https://cdn-images-1.medium.com/max/800/1*hnBzd86OgPXHsF7rV0tAcQ.png)

This system of calculating partial derivatives by working backwards is known as [backpropagation](https://en.wikipedia.org/wiki/Backpropagation).We'll use [Stochastic-Gradient-descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) to update the weights , bias to minimize the loss.

![](https://cdn-images-1.medium.com/max/800/1*kX2Av8AoG8VX42kXhhFHZw.png)

*η*  is a constant called the  **learning rate** that controls how fast we train.

All we’re doing is subtracting *η*∂*w*1/​∂*L*​ from *w*1​.

so that w1 is updated with the new w1.

Our training process will look like this:

1. Choose **one** sample from our dataset. This is what makes it *stochastic*gradient descent — we only operate on one sample at a time.
2. Calculate all the partial derivatives of loss with respect to weights or biases (e.g. ∂*w*1/​∂*L*​, ∂*w*2​/∂*L*​, etc).
3. Use the update equation to update each weight and bias.

As we train the NN more and more , our loss will get more and more accurate this way.

![](https://cdn-images-1.medium.com/max/800/1*meeIavVtb0G0hNF6UvkkGw.png)

Resources :

https://learn-neural-networks.com/single-layer-neural-network-training/
