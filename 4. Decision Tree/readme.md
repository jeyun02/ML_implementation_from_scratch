# 4. Decision Tree.

ML > Classification > Supervised Learning  
 
 트리의 각 노드는 특징(feature)과 임계값(threshold)을 기반으로 데이터를 분할하고, 최종적으로 leaf 노드에서 예측값을 제공한다.
 정보 이득(Information Gain)을 기반으로 최적의 분할을 찾는 것이다.
## 정보 이득(Information Gain) 계산

정보 이득은 부모 노드의 엔트로피와 자식 노드들의 가중 평균 엔트로피의 차이로 계산된다.  정보 이득이 클수록 해당 분할이 데이터를 더 잘 분류한다는 것을 의미한다

1. **엔트로피(Entropy):**  데이터 집합의 불순도를 측정하는 지표다.  엔트로피가 높을수록(heterogeneous 할수록) 데이터가 혼잡하게 섞여있다는 것을 의미한다

   수식:
   $H(Y) = -\sum\limits_{i=1}^{c} p_i \log_2 p_i$

   - $Y$: 목표 변수 (Target Variable)
   - $c$: 클래스의 개수
   - $p_i$: i번째 클래스의 비율   
   
   이때 Entrophy 값을 비교만 할 때는 $log_2 $ 대신 $log$를 써도 무방하여, DecisionTree.py 에서는 $log $를 사용했다.

2. **자식 노드 엔트로피:** 특정 특징과 임계값을 기준으로 데이터를 분할했을 때, 자식 노드들의 엔트로피를 계산한다.

   수식:
   $H(Y|X) = \sum\limits_{j=1}^{k} \frac{N_j}{N} H(Y_j)$

   - $X$: 특징 (Feature)
   - $k$: 분할된 자식 노드의 개수 (이 코드에서는 2개 - 왼쪽, 오른쪽)
   - $N_j$: j번째 자식 노드의 데이터 개수
   - $N$: 전체 데이터 개수
   - $H(Y_j)$: j번째 자식 노드의 엔트로피

3. **정보 이득:** 부모 노드의 엔트로피와 자식 노드들의 가중 평균 엔트로피의 차이입니다.

   수식:
   $IG(Y, X) = H(Y) - H(Y|X)$

**코드 설명:**

- `_information_gain()` 함수: 정보 이득을 계산한다.
- `_entropy()` 함수: 엔트로피를 계산한다.
- `_split()` 함수: 특징과 임계값을 기반으로 데이터를 왼쪽 자식 노드와 오른쪽 자식 노드로 분할한다.
- `_best_split()` 함수:  가능한 모든 특징과 임계값 조합에 대해 정보 이득을 계산하고, 정보 이득이 가장 큰 특징과 임계값을 선택한다.
- `_grow_tree()` 함수: 재귀적으로 트리를 생성한다. 정지 조건을 만족하면 leaf 노드를 생성하고, 그렇지 않으면 `_best_split()` 함수를 사용하여 최적의 분할을 찾고 자식 노드를 생성한다.

**예시:**

만약 목표 변수 Y가 [0, 0, 1, 1, 1] 이라고 가정하면,

- 부모 노드 엔트로피:
$H(Y) = -(\frac{2}{5}\log_2\frac{2}{5} + \frac{3}{5}\log_2\frac{3}{5}) \approx 0.97$

특정 특징 X로 분할 후, [0, 0] 와 [1, 1, 1] 로 나뉘었다면,

- 왼쪽 자식 노드 엔트로피: $H(Y_1) = 0$ (모두 0)
- 오른쪽 자식 노드 엔트로피: $H(Y_2) = 0$ (모두 1)
- 자식 노드 엔트로피: $H(Y|X) = \frac{2}{5} * 0 + \frac{3}{5} * 0 = 0$
- 정보 이득: $IG(Y, X) = 0.97 - 0 = 0.97$

이처럼 정보 이득이 최대화되는 feature와 threshold을 찾아 트리를 구축한다.[[1]](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FwiPMw%2FbtrzKQIxL5C%2FUU4JCaX9S2CIeyvS3tnmfK%2Fimg.png)[[2]](https://gofo-coding.tistory.com/entry/Decision-Tree-Information-Gain-1)


## 함수 호출 흐름도 그리고 싶은데
<!-- TODO 
data flow 를 시각화 해보고 싶은데.
-->



## 키워드 전용 인자.

<!-- 
? 키워드 전용 인자.
[o] 키워드 전용 인자.
-->

DecisionTree.py 를 보면 Node.__init__() 의 인자에 *(asterisk) 가 들어있다. 무슨 의미일까.

```python
class Node:
    def __init__(self, feature, threshold, left, right,*, value, ):
   
```

키워드 인자, 위치 인자, 키워드 전용 인자, 위치 전용인자.[1]

위치인자란 함수 호출 시 인자들의 위치로 그 값을 인식해서 위치인자 값에 저장할 수 있는 인자.
반면 키워드 인자는 함수 호출 시 '인자=값'처럼 위치 키워드 값을 넣어 주어야 하는 인자.

저 인자들 중 * 뒤에 오는 인자들(여기서는 value)는 키워드 전용 인자임.
비슷하게 * 대신 *args를 쓰면 뒤에오는 인자들은 키워드 전용 인자로 구분 하되,
위치 인자를 함수 정의 이후 추가할 수 있다는 장점이 있음.


## np.argwhere() 
<!-- 
? np.argwhere()
[o] np.argwhere()
-->

np.argwhere(조건식) 설명   
조건식에 해당하는 array의 값의 위치 idx 들을 array로 나열하여 반환해준다.

```python
import numpy as np
x =[10,20,30,40,50]   
np.argwhere(x<40)   

>>> array([[0],[1],[2]])
```

# References
[[1] 참고 이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FwiPMw%2FbtrzKQIxL5C%2FUU4JCaX9S2CIeyvS3tnmfK%2Fimg.png).   
[[2] 참고 link](https://gofo-coding.tistory.com/entry/Decision-Tree-Information-Gain-1)
[[3] 키워드 인자 설명](https://daryeou.tistory.com/386)