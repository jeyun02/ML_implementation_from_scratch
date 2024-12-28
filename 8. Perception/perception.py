import numpy as np

def unit_step_func(x):
   return np.where(x >= 0, 1, 0)

class Perception:
   
   def __init__(self, learning_rate=0.1, n_iters=1000):
      
      self.learning_rate = learning_rate
      self.n_iters = n_iters
      self.activation_func = unit_step_func
      self.weights = None
      self.bias = None

   def fit(self, X, y):
      pass

   def predict(self, X):
      pass
