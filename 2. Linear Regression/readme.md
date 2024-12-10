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

생각해본 다른 아이디어가 있긴 하다.   
1. y의 최대 최소 범위를 x와 동일하게 만들기 위해 y에 가중치y_norm를 곱하여 확대 또는 축소한다.(b까지 하려면 y평균만큼 평행이동 시키는 것도 방법일 듯.)
2. 이후 선형회귀분석을 실시하여 w 와 b 를 구한다.
3. y에 대한 함수의 coefficient 로써 w 와 b 가 존재하기 때문에 y_norm 만큼 다시 나눠 주게 되면 원래 범위의 y 와 x 에 대한 선형 회귀 함수가 나올 수 있다.
4. 새로운 모델 ( w_new, b_new) 를 가지고 x_test 를 입력해준다.

이러면 lr 을 굳이 1 로 바꾸고, w 와 b 의 초기값을 변경하지 않아도 되지 않을까.