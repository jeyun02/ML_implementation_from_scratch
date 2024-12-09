# Linear Regression
<center>
단순 선형 회귀 최종 모델 
</center>

$$
\hat{y} = w_1 x + w_0
$$ 

## 변수 설명.
$y$ 는 정답값으로써의 $y$를 의미한다.    
$\hat{y}$ 는 최적화된 함수의 함수값으로써의 $y$를 의미한다.   
$y_{pred}$ 는 최적화된 함수에 $ x_{test}$를 대입한 최종 결과값으로써의 $y$를 의미한다. 

목표는 최적의 $ w_1 $ 과 $ w_0 $ 을 구해 $\hat{y} $와 $x$ 에  대한 함수를 만드는 것이다.   

구한 $ \hat{y} = w_1 x + w_0$ 최종함수에 $ x_{test}$ 를 대입하면 구체적인 값인 $ y_{pred} $ 가 도출된다.


## 함수 최적화 과정.
함수를 최적화한다는 것은 곧$w_1$ 과 
$w_0$를 최적화하는 것과 같은 말이다.
### MSE 를 COST 함수로.
$$  COST(w_1, w_0) = TODO $$ 




$w_1$ 과 $COST$,   
$w_0$ 과 $COST$   
각각을 함수로 표현할 수 있고, 그 함수들을 $w_1$ 과 $w_0$ 로 미분할 수도 있다.

최적의 $w_1, w_0$ 에 가까이 다가가는 함수를 만든다.

$w_1$ 를 최적화(optimize)하는 함수는 이러하다.

