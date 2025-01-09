# 3. Logistic Regression   


로지스틱 함수(시그모이드 함수) + 선형회귀 = 로지스틱 회귀.

## 변수 정의
$y = \{y  | y= 0,1 \}$   
$\vec{x} = [1, x_1, ..., x_n]$  
$\vec{\beta} = [\beta_0, \beta_1, ..., \beta_n]$

## 로지스틱 회귀 함수 변환 과정 유도.
###  1. 기본 선형회귀함수에서 시작한다.

$$\hat{y} =  \vec{\beta}^T\vec{x} +\varepsilon$$
## 2. 좌변 변형. 이 식에서 선형성이 유지된다는 가정하에 좌변을 변형시킨다.[1]

$y  \cdots (y ∈ \{0, 1\})
\\ \mid\\
\\ \small{(1)}\\ 
\\ \downarrow\\
p(x) = P(Y=1|X=\vec{x}) \cdots (p(x) ∈ [0,1])
\\ \mid\\
\\ \small{(2)}\\ 
\\ \downarrow\\
odds = \frac {P(Y=1|X=\vec{x})} {1−P(Y=1|X=\vec{x})} \cdots (odds ∈ [0,\infty))
\\ \mid\\
\\ \small{(3)}\\ 
\\ \downarrow\\ 
log(odds) = log(\frac {P(Y=1|X=\vec{x})} {1−P(Y=1|X=\vec{x})}) = \frac{p(x)}{1−p(x)} \cdots (log(odds) ∈ (-\infty,\infty))$ 

(1)  
 $y$ 가 범주형이기 때문에 연속형 변수로 바꿔주기 위해 $y$를 $y$가1 일 확률 $p(x)$로 변형시킨다. (선형성은 유지된다.)

(2)  
$p(x)$ 와 $odds(p(x))$ 사이에 선형성이 유지된다고 가정.

(3) $ $   
$odds(p(x))$ 와 $log(odds(p(x)))$ 사이에 선형성이 유지된다고 가정.

즉, 로지스틱 회귀를 사용할 때, 로지스틱함수와 독립변수들간에 선형성이 유지된다는 가정이 필요하다.

## 3. $p(x) $에 대한 함수로 정리
이제 $p(x)$ 에 대해 풀어보면,
$p(x) = \frac{1}{1 + e^{-\vec{\beta}^T\vec{x}}}$


# 4. Optimization: cost 는 binary cross entropy.

linear regression 때와 마찬가지로.


$\vec{\beta}$ 를 최적화(optimize)하는 함수는

$\vec{\beta} = \vec{\beta} - \alpha \frac {dCE(\vec{\beta})}{d\vec{\beta}}$   
이며,
계산과정을 생략하면,   [2]
 $\frac {dCE(\vec{\beta})}{d\vec{\beta}}  = \frac1n \sum\limits_{i=1}^n (\hat{y} - y_i)x_i$

# 5. 최종 예측
$$
y_{pred} = \begin{cases}
1 ,& \text{if }p(x)> 0.5 \\
0, & \text{if }p(x)\leq0.5
\end{cases}
$$

# References
[1] https://ratsgo.github.io/machine%20learning/2017/04/02/logistic/
[2] https://plan0a-0z-entering-security.tistory.com/110
