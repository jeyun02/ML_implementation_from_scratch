
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

X, y = datasets.make_regression(
   n_samples=100, 
   n_features=1, 
   noise=20, 
   random_state=4
   )
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

# X,y 를 시각화하여 보자.
# 시각화 결과 X의 분포는 -3에서 3까지 증가하고 정도인 반면
# y의 분포는 -200 에서200 까지 증가하는 범위로 
# 눈으로 봤을 때 weight가 대략 50정도 나와야 정상이지만 디버깅 결과 근접하지 못하는 모습을 보임.
# 이럴 때는 정규화를 해서 regression을 한다음 함수를 역 정규화해야하나????
"""
fig = plt.figure(figsize= (8,6))
plt.scatter(
   X[:,0],
   y,
   color="b",
   marker="o",
   s=30
   )
plt.show()
"""
reg = LinearRegression()
reg.fit(X_train,y_train)
predictions = reg.predict(X_test)

def MSE(y_test, predictions):
   return np.mean((y_test - predictions) **2)

mse = MSE(y_test, predictions)
#print(mse)
#print("predictions: ", predictions)
#print("weight: ", reg.weights)
#print("bias: ", reg.bias)


# 최종 결과 시각화.
y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
# 최종 모델 함수
# 이게 왜 일직선으로 되냐면 prediction 할 때 1차 함수를 통과하기 때문에 모든 X 와 y의 요소들은 한 직선 위에서 대응함. 
# pyplot.plot 은 그 일직선 상 점들을 이어주는 역할만 할 뿐임. 함수자체를 이해해서 표현하는 게 아니라.
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction') 
plt.show()