# 6. Naive Bayes

ML > Classification > Gaussian Naive Bayes

### argmax 
argmax(f(x)) 란.
f(x) 를 최대로 만들어주는 x 값을 말한다.
여기에서 y_pred = argmax_y(P(y|X)) 이고,
즉 argmax 함수 자체가 predicted label 을 제출하는 핵심 함수로 사용된다.


## 데이터 설명.
```python
from sklearn import datasets
X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
```

sklearn.datasets.make_classification 으로 분료 데이터셋을 생성한다.[[scikit learn, make_classification]](https://scikit-learn.org/dev/modules/generated/sklearn.datasets.make_classification.html)

데이터 분포는 std =1 인 정규분포로 생성된다.
반환은 X, y 를 ndarray 형식으로 반환한다.

> X.shape = (1000,10)   
y.shape = (1000,1)
d

## 함수 설명.

$ P(Y | X) = P(Y \mid x_1, \dots, x_{10}) =  $   

$X = [x_1,x_2,...,x_{n=10}]$ 값을 입력했을 때 $Y=y_{label}$가 나올 확률을 계산함으로써 prediction의 fitting 없이 수학적 예측 알고리즘의 역할을 함.

$ P(Y_c \mid X_1, \dots, X_n) $ 를 구하는 것이 Naive Bayes 알고리즘의 목표임.   
여기서 "Naive"란 대충대충 이라는 뉘앙스가 있음.   
다시 말해, $X_i$들이 독립적이라는 강력한 가정을 Naive 하게 사용했다는 의미임.   
이 가정을 바탕으로 위 $ P(Y_c \mid X_1, \dots, X_n) $ 를 계산하기 편하게 변환함.

### 1. 식정리: Bayes Theorem 을 이용
>$P(Y \mid x_1, \dots, x_n) =\frac{P(Y) P(x_1, \dots, x_n \mid Y)}{P(x_1, \dots, x_n)}  \cdots\text{(1) 베이즈 정리 사용 }$   

### 2. 식변형: 상수는 영향을 안준다.
>$\frac{P(Y) P(x_1, \dots, x_n \mid Y)}{P(x_1, \dots, x_n)} \\ \downarrow \cdots \text{분모 상수취급}\\ 
P(Y) P(x_1, \dots, x_n \mid Y)$

### 2. 식정리: 독립임을 이용
$A, B$가 독립인 경우 다음이 성립한다.   
>$ P(A \cap B) = P(A|B)P(B) =P(A)P(B) $   
>$P(x_1, \dots, x_{10} \mid Y) =P(x_1 \cap x_2\dots\cap x_{10} \mid Y)=
P(x_1|Y)P(x_2|Y)\cdots P(x_{10}|Y)= 
\prod_{i=1}^{n=10} P(x_i \mid Y)$   

즉,   
> $ \frac{P(Y) P(x_1, \dots, x_n \mid Y)}{P(x_1, \dots, x_n)} = \frac{P(Y) \prod_{i=1}^{n=10} P(x_i \mid Y)}{\prod_{i=1}^{n=10} P(x_i \mid Y)}\cdots(2) Naive 독립 적용$

### 3. 식변형: log 함수 적용 
확률들의 곱으로 이루어져서 수가 너무 작아지므로, log 함수를 이용해 덧셈으로 변환해보자.(log함수는 (0,1)범위에서 단조증가함수이므로 log 적용해도 함수 개형이 유지된다.)

> $\prod_{i=1}^{n=10} f(x_i) 
>\\ \downarrow \cdots \text{log(x) 적용}\\
>log(\prod_{i=1}^{n=10} f(x_i)) = \sum\limits_{i=1}^{n=10} log(f(x_i))$

이 변형을 적용하면,

>$ P(Y) \prod_{i=1}^{n=10} P(x_i \mid Y)
>\\ \downarrow \\ 
>log(P(Y) \prod_{i=1}^{n=10} P(x_i \mid Y)) = log(P(Y)\sum\limits_{i=1}^{n=10} P(x_i \mid Y))$

### 4. 식변형: argmax 적용 후 식정리
x를 입력했을 때 label이 나올 확률(P(Y|X)) 를 구하는 함수를 알기 때문에. 가장 나올 확률이 높은 label(y)값을 출력하기 위해 argmax 를 적용하자.


> $ P(Y|X) = P(Y \mid x_1, \dots, x_n) \\
> \downarrow \cdots 1., 2., 3. 적용 후\\
> P(Y|X) \approx log(P(Y)\sum\limits_{i=1}^{n=10} P(x_i \mid Y))\\
> \downarrow \cdots argmax 적용\\
> \hat{y} \\
> = argmax_y{P(Y|X)} \\ 
> \approx argmax_y{log(P(Y)\sum\limits_{i=1}^{n=10} P(x_i \mid Y))} \\= argmax_y{log(P(x_1|y)) + log(P(x_2|y)) + \cdots log(P(x_n|y)) + log(P(y))}$ 

### 5. Gaussian PDF for $P(x_i \mid y)$
$P(y) =\frac{n(Y=y)} {n(Y)}$ : 전체 Y 개수 중 Y=y인 것의 비율 

$P(x_i \mid y) = \frac{1}{\sqrt{2 \pi \sigma_y^2}} 
\exp\left( -\frac{(x_i - \mu_y)^2}{2 \sigma_y^2} \right)$ : Y=y 일 때, x 들의 분포함수 (PDF 함수)

e.g.   
$y$가 1이라고 주어졌을 때, $x_1$ column에 대한 $\mu_y,\sigma^2_y$를 구할 수 있음.   
$y$가 1이라고 주어졌을 때, $x_1$ column의 값들에 대한 분포를 Gaussian 정규분포로 가정...

### 6. 함수들 최종 조합

이제 이 4.의 함수들을 조합해주면,
```python
prior = np.log(self._priors[idx])
posterior = np.sum(np.log(self._pdf(idx, x)))
posterior += prior
```


? classification dataset을 생성할 때 default 로 gaussian 분포를 띠게 생성한 건 알겠는데, Naive Bayes 에서 gaussian 분포를 가정하는 특별한 이유가 있는지?



# References

1. [scikit-learn, Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)

