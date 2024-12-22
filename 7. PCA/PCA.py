import numpy as np

class PCA:
   def __init__(self, n_features):
      self.n_features = n_features
      self.mean = None
      

   def fit(self,X):
      """ No need of y beause PCA is unsupervised learning."""
      
      # centering X
      self.mean = np.mean(X, axis=0)
      X = X - self.mean
      
      # cov(X,X)
      cov = np.cov(X.T)

   
   def predict(self,X):
      pass
