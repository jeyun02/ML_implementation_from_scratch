
import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
   distance = np.sqrt(np.sum((x1-x2)**2))
   return distance

class KNN:
   def __init__(self, k=3):
      self.k = k

   def fit(self, X, y):
      self.X_train = X
      self.y_train = y

   def predict(self, X):
      predictions = [self._predict(x) for x in X]
      return predictions
   
                  #   [1,1,1,1,1,1,1 ... 1 ] 
                  #            X 개
   def _predict(self, x):
      # compute the distance(유클리디안 거리)
      distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
      
      # get the closest k y labels 
      # >>> np.argsort([20, 49, 29])
      # >>> [0,2,1] # 값기준으로 정렬 된 indices 를 반환.
      k_indices = np.argsort(distances)[:self.k]         
      k_nearest_labels = [self.y_train[i] for i in k_indices]
      
      # majority vote
      #[a,a,b] 가 주어졌을 때
      # 가장 많은 요소인 a 를 반환하는 함수 -> collections 의 counter 를 사용한다.
      most_common = Counter(k_nearest_labels).most_common()
      return most_common[0][0]
   """

import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    
        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
