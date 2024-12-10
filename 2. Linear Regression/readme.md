# Linear Regression
<center>
단순 선형 회귀 최종 모델 
</center>

$$
\hat{y} = w x + b
$$ 

## 변수 설명.
$y$ 는 정답값으로써의 $y$를 의미한다.    
$\hat{y}$ 는 최적화된 함수의 함수값으로써의 $y$를 의미한다.   
$y_{pred}$ 는 최적화된 함수에 $ x_{test}$를 대입한 최종 결과값으로써의 $y$를 의미한다. 

목표는 최적의 $ w $ 과 $b $ 을 구해 $\hat{y} $와 $x$ 에  대한 함수를 만드는 것이다.   

구한 $ \hat{y} = w_1 x + w_0$ 최종함수에 $ x_{test}$ 를 대입하면 구체적인 값인 $ y_{pred} $ 가 도출된다.


## 함수 최적화 과정.
함수를 최적화한다는 것은 곧$w$ 와 $b$를 최적화하는 것과 같은 말이다.
### MSE 를 MSE 함수로.
$$  MSE(w, b) = TODO $$ 




$w$ 과 $MSE$,   
$b$ 과 $MSE$   
각각을 함수로 표현할 수 있고, 그 함수들을 $w_1$ 과 $w_0$ 로 미분할 수도 있다.


여기서    
$\frac d{dw} MSE(w,b)  $를 코드 내에서 $ dw$ 라고 하고  
$\frac d{db} MSE(w,b)  $를 코드 내에서 $ db$ 라고 하자.

실제 미분을 통해 dw와 db 를 구하면,   
$dw = \frac 2n \Sigma (y_{pred} - y)x  $   
$db = \frac 2n \Sigma (y_{pred} - y)  $
이다.    

dw 를 구할 때 test.py 에서    
`dw = (1/n_samples) * np.dot(X.T, (y_pred - y))`
로 np.dot 연산만 사용했는데, np.dot 자체가 마지막에 np.sum 까지 자동으로 실행한다고 한다.

변화량을 구했으니 w를 갱신해야한다.

최적의 $w, b$ 에 가까이 다가가는 함수를 만든다.

$w$ 와 $b$를 최적화(optimize)하는 함수는 이러하다.

$ w = w - \alpha dw $   
$ b = b - \alpha db $

이상이다. 끝! 이렇게 최적의 w 와 b 를 구한 후 x_test 를 입력하면 y_test 가 나온다. 이게 linear regression!

# learning rate 와 초기값 에 대한 고민
test.py 에서 linear regression 데이터를 생성했을 때 (x,y) 좌표평면을 시각화 했다. 대략 (-3,-200) 에서 (3,200) 까지 우 상향하는 것을 보야 약 50이상의 기울기w가 나와야 할 것으로 예상 됐다. 그러나, reference 상에서는 lr 을 0.001 로 설정해서 도저히 w 가 75에 도달할 기미가 안보였다. 그래서 임의로 w 의 초기값을 50으로 설정했다. `self.weights = np.zeros(n_features) + 50` 그리고 lr 의 초기값도 1로 최종 선택했다.   


## 정규화 및 표준화가 가능하다.
생각해본 다른 아이디어가 있긴 하다.   
1. y의 최대 최소 범위를 x와 동일하게 만들기 위해 y에 가중치y_norm를 곱하여 확대 또는 축소한다.(b까지 하려면 y평균만큼 평행이동 시키는 것도 방법일 듯.)
2. 이후 선형회귀분석을 실시하여 w 와 b 를 구한다.
3. y에 대한 함수의 coefficient 로써 w 와 b 가 존재하기 때문에 y_norm 만큼 다시 나눠 주게 되면 원래 범위의 y 와 x 에 대한 선형 회귀 함수가 나올 수 있다.
4. 새로운 모델 ( w_new, b_new) 를 가지고 x_test 를 입력해준다.

이러면 lr 을 굳이 1 로 바꾸고, w 와 b 의 초기값을 변경하지 않아도 되지 않을까.
```python
from sklearn.preprocessing import MinMaxScaler

# 정규화를 위한 scaler 객체 생성
scaler = MinMaxScaler()

# X_train 정규화 및 변환
X_train_normalized = scaler.fit_transform(X_train)

# X_test는 X_train의 스케일을 사용하여 변환
X_test_normalized = scaler.transform(X_test)

# y에 대해서도 동일한 과정 적용
y_scaler = MinMaxScaler()
y_train_normalized = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_normalized = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
```
```python
from sklearn.preprocessing import StandardScaler

# 표준화를 위한 scaler 객체 생성
scaler = StandardScaler()

# X_train 표준화 및 변환
X_train_standardized = scaler.fit_transform(X_train)

# X_test는 X_train의 스케일을 사용하여 변환
X_test_standardized = scaler.transform(X_test)

# y에 대해서도 동일한 과정 적용
y_scaler = StandardScaler()
y_train_standardized = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_standardized = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
```
```python
# 예측 결과를 원래 스케일로 되돌리기
y_test_original = scaler.inverse_transform(y_test_normalized)
y_test_original = scaler.inverse_transform(y_test_standardized)
```

## 다른 방법. perplexity 질문.
이러한 상황에서 학계에서 일반적으로 사용하는 해결 방법은 다음과 같습니다:

1. 학습률 조정: 초기 학습률을 증가시켜 더 빠른 수렴을 유도합니다. 예를 들어, 0.001에서 0.01 또는 0.1로 증가시킵니다[1][4].

2. 반복 횟수 증가: iterations를 1000에서 더 큰 값(예: 5000 또는 10000)으로 늘려 모델이 충분히 학습할 시간을 제공합니다[1].

3. 특성 스케일링: 입력 데이터를 정규화하거나 표준화하여 모든 특성이 비슷한 스케일을 가지도록 합니다. 이는 경사 하강법의 수렴 속도를 높일 수 있습니다[1][2].

4. 학습률 스케줄링: 학습 과정에서 학습률을 동적으로 조정합니다. 예를 들어, 지수 기반 스케줄링이나 1 사이클 스케줄링을 사용할 수 있습니다[2][4].

5. 모멘텀 사용: 경사 하강법에 모멘텀을 추가하여 최적화 과정을 가속화하고 지역 최솟값에 빠지는 것을 방지합니다[4].

Citations:
[1] https://ai-bt.tistory.com/entry/%EC%84%A0%ED%98%95-%ED%9A%8C%EA%B7%80-Linear-Regression-2-%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95-%EB%8D%B0%EC%9D%B4%ED%84%B0-%ED%91%9C%EC%A4%80%ED%99%94
[2] https://hwk0702.github.io/ml/dl/deep%20learning/2020/08/28/learning_rate_scheduling/
[3] https://yhyun225.tistory.com/9
[4] https://box-world.tistory.com/70
[5] https://realblack0.github.io/2020/03/27/linear-regression.html
[6] https://www.startupcode.kr/139394a4-8061-80c0-86b3-cc64ba911318
[7] https://aiclaudev.tistory.com/22
[8] https://www.purestorage.com/kr/knowledge/what-is-learning-rate.html
[9] https://velog.io/@wkfwktka/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EC%84%A0%ED%98%95-%ED%9A%8C%EA%B7%80
