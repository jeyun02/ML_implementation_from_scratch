import numpy as np

def sigmoid(a):
   # odds 를 구한다. ( 생략)
   # 로짓변환을 한다.(생략)
   return 1/(1 + np.exp(-a))
   # np.exp() 를 사용할 때 수가 너무 커져 오버플로우가 발생한다. 젠장!
   # -a 를 np.float128(-a) 로 변경해보자 -> 64 비트 운영체제에서는 안되나본데? anaconda 같은거 사용해야하나봐
   
   # 어떤 블로그에서는 exp(a - np.max(a))를 해보라는데... 그럼 왜곡이 생기지 않나? 일단 해보자.
   # return 1/(1 + np.exp(-a - np.max(-a))) -> 으로 변경했을 때 accuracy 가 0.92에서 0.61로 폭락했다.

class LogisticRegression:
   def __init__(self, lr=0.01, n_iters=1000):
      # 단순 로지스틱회귀의 변수 정의
      self.lr = lr
      self.n_iters = n_iters
      self.w = None
      self.b = None

   def fit(self, X, y):
      # sample 수와 feature 수를 X.shape 로 바로 구할 수 있음.
      n_samples, n_features = X.shape
      # 초기변수 값들을 설정한다.
      self.w = np.zeros(n_features)
      self.b = 0

      for _ in range(self.n_iters):
         linear_predictions = np.dot(X, self.w) + self.b
         predictions = sigmoid(linear_predictions)
         
         # 이후 gradient descent 과정을 거쳐야 하므로
         # cross entrophy 에 대한 d/dw 와 d/db 를 구현하자!
         dw = (1/n_samples) * np.dot(X.T, predictions - y)
         db = (1/n_samples) * np.sum(predictions - y)

         # w 와 b 를 갱신하는 gradient 방식은 언제나 동일하다.
         # w:= w - lr * dw 
         self.w = self.w - self.lr * dw
         self.b = self.b - self.lr * db
      
   def predict(self, X_test):
      # 마지막 결과 예측과정: x 에 w 와 b 가 최적화된 sigmoid 를 적용한다. 
      # 그 sigmoid 값이 1/2 보다 크면 yes 아니면 no 로 분류
      
      linear_pred_final = np.dot(X_test, self.w) + self.b
      pred_final = sigmoid(linear_pred_final)
      class_pred_final = [1 if y >0.5 else 0 for y in pred_final]
      # linear_predictions_final > 0.5 이면 1 로 판단. array 를 반복자로 반복하여 새로운 array 를 만든다.
      return class_pred_final
