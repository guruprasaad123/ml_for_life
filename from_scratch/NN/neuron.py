import numpy as np

def sigmoid(x):
  # Our activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

class Neuron:
  def __init__(self, weights, bias,inputs):
    self.weights = weights
    self.bias = bias
    self.inputs=inputs

  def sum(self):
    # Weight inputs, add bias, then use the activation function
    total = np.dot(self.weights, self.inputs) + self.bias
    return total
        
        

  def feedforward(self):
    # returning the sigmoid function this is used as Activation function 
    total = self.sum()
    return sigmoid(total)

