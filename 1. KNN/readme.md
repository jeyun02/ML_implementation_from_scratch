# KNN
Full name: K - nearest neighbors   
한글 : k-최근접 이웃 알고리즘   

ML > Classification > Supervised learning

## test.py

데이터는 scikit-learn > database > iris 를 사용한다.

### 데이터 형식
``` python 
X = [   
   [int,int,int,int],   
   [int,int,int,int],   
   [int,int,int,int],   
   ...   
   [int,int,int,int],   
   ]   

y = [int,int,int,...int]
```
- shape of X =    (150 , 4)   
- shape of X_train =    (120 , 4)   
- shape of X_test = (30 , 4)   
- shape of y = (150 , 1)   
- shape of y_train = (120 , 1)   
- shape of y_test = (30 , 1)   


## KNN.py
### 거리 계산.
거리 계산은 euclidean 거리를 사용하므로
만약 각 차원에 대해 정규화를 하지 않는다면?
간격이 좁게 설정된 차원에 대해서 민감하게 반응하게 된다.   
이런 차이가 예측에 도움이 될까? 아니면 정규화로 제거해야 할 현상일까?


### fit이 없다?
KNN.py 에 보면 

```python
def fit(self, X, y):
   self.X_train = X
   self.y_train = y
```
fit 부분이 생략되어 있다.
knn 자체는 학습과정이 없다. 즉 모델을 ground truth에 맞게 최적화시키는 fitting 과정 없이 바로 predict한다.   
따라서 KNN 은 ML 모델이라기보다 ML 알고리즘의 성격에 더 가깝다.

그러나 학습(fitting)할 수 있는 parameter가 하나 있다. 바로 임의로 설정했던 k이다.   
knn에는 최적의 k 를 학습하는 과정이 필요하다.   
대표적으로 K-cross fold

연관해서 test.py 의 X_test 는 정확히 X_validation 의 역할을 한다.


흔히 ML 모델 중 가장 간단하다는 부분이 바로 이런부분에서 인듯하다.





 # references  
[How to implement KNN from scratch with Python - AssemblyAI](https://www.youtube.com/watch?v=rTEtEy5o3X0&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=2)

## 추가 구현할 수 있는 내용

1. cross fold
2. X 정규화.


## After Questions

1. np.argsort 와 sort 의 차이.
2. Collections 의 Counter 를 import 할수 없는 상황이 분명 있을 거 같은데.
