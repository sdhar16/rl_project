import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

def choose_action(q_vals, allowed_action, epsilon):  
  q_valid = np.copy(q_vals * allowed_action)
  q_valid[allowed_action==0] = -np.inf

  if(np.random.random() > epsilon and np.max(q_valid)>-np.inf):
    action =  np.argmax(q_valid)
  else:
    a_, _ = np.where(allowed_action==1)
    action = np.random.permutation(a_)[0]

  return action

def moving_average(data, window):
  data = np.array(data)
  weights = np.array([2. ** i for i in range(window//2+1)] + [2. **i for i in range(window//2+1,-1, -1)])
  weights /= np.sum(weights)
  data = np.convolve(data, weights, 'valid')
  return data

def plot(data, title, xlabel, ylabel, c, smooth = 101, exp = False):

  if(exp):
    data = moving_average(data, smooth)
  else:
    data = np.convolve(data, np.ones(smooth)/smooth, mode='valid')
  
  plt.plot(data, c+"-")
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)

  plt.show()






