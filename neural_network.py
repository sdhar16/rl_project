from lib2to3.pgen2 import grammar
import numpy as np
import numpy.matlib as matlib

np.random.seed(42)

class NeuralNetwork:
  def __init__(self, input_size, output_size, hidden_size=200, learning_rate = 0.01):

    self.lr = learning_rate

    self.hidden_layer = np.zeros((output_size, hidden_size))

    self.w1 = np.random.normal(0,1,(hidden_size, input_size+1)) * np.sqrt(1/input_size+1)
    self.w2 = np.random.normal(0,1,(output_size, hidden_size)) * np.sqrt(1/hidden_size)

  def forward(self, x):
    a = self.w1 @ x
    h = self.relu(a)
    y = self.w2 @ h

    return y, h

  def get_error(self, y, t):
    return np.linalg.norm(y - t)
  
  def relu(self, x, derivative=False):
    if(derivative):
      return 1.0 *  (x>0)
    return x * (x>0)

  def gradient(self, y, t, h, x):
    g2 = (t - y) @ h.T
    g1 = ((self.w2.T @ (t - y)) * self.relu(h, derivative=True)) @ x.T
    return g1, g2

  def backword(self, y, t, h, x):
    g1, g2, = self.gradient(y, t, h, x)

    self.w1 += -self.lr * g1
    self.w2 += -self.lr * g2
  
  def fit(self, y, target , h, x):
    self.backword(y, target, h, x)

