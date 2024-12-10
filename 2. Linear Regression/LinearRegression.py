from heapq import nsmallest
import numpy as np


class LinearRegression:
   """ 여기서 learning rate는 0.001, 0.01 0.1 1 등 다양하게 사용되는데 
       시각화를 해보니 기울기가 대략 50이 넘을 거 같고,
       0.001의 lr으로는 mse가 줄어들지 않으며 iteration이 부족하게 느껴져서 
       lr을 1로 수정하게 됨.
       아니면 weights 의 초기값을 한 70정도로 해놓고 시작하는 것도 나쁘지 않음.
       ㄴ 이렇게 적절한 초기 값을 수학적으로 설정하면 좋긴 할 듯."""
   def __init__(self, lr =1, n_iters =1000):
   
      self.lr = lr
      self.n_iters = n_iters
      self.weights = None 
      self.bias = None
      
   def fit(self,X, y):
      ## Optimize the function
      n_samples, n_features = X.shape # (120,1) 이라고 해보자
      self.weights = np.zeros(n_features) + 50 # 초기값을 50으로 두고 시작
      self.bias = 0

      for _ in range(self.n_iters):
         # Optimize:
         # X는 유지되고 self.weights, self.bias 는 n_iters 번 갱신됨.
         y_pred = np.dot(X, self.weights) + self.bias
         
         # dw, db는 각각 MSE cost 함수를 weight와 bias로 미분한 것. 
         dw = (1/n_samples) * np.dot(X.T, (y_pred - y)) # np.dot 에 np.sum 기능까지 포함되어있대... 뭐지.
         db = (1/n_samples) * np.sum(y_pred - y)

         # weight 와 bias 갱신.
         self.weights = self.weights - self.lr * dw
         self.bias = self.bias - self.lr * db
   
   def predict(self, X):
      # fit 에서 구한 함수에 그대로 대입한다.
      y_pred_final = np.dot(X,self.weights) + self.bias
      return y_pred_final