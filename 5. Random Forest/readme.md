# 5. Random Forest.

랜덤 포레스트란.
여러 개의 의사결정 나무의 결과를 종합해서 분류 및 회귀를 실행하는 모델.
의사결정 나무 생성 알고리즘 자체는 동일하되, 대신 모집단의 크기만큼 무작위복원추출을 반복하여
서로 다른 의사결정 나무들을 생성한다.   
그 결과를 bagging boosting stacking 등으로 분산을 줄일 수 있다고 함.


## Bootstrap.

부트스트래핑이란, 크기가 n인 모집단을 같은 크기로 무작위 복원추출하는 것을 반복하여 통계량을 추정하는 방법.

> "부트스트래핑"이라는 이름은 "자기 신발끈으로 자신을 끌어 올리기(pull oneself up by one's own bootstraps)"이라는 속담에서 유래했습니다.이 말은 컴퓨터의 전원을 켜는 "부팅"의 어원이기도 합니다. 왜 이런 이름이 붙었냐하면, 부트스트래핑이 데이터의 분포에 대해 가정하지 않고 통계량을 추정하는 방법이기 때문입니다.(MINDSCALE, 2022.02.21)


### np.swapaxes() 사용.
```python
import numpy as np
# np.swapaxes() 활용
x = np.array([
   [[0,0,0,0],[1,1,1,1],[2,2,2,2]],
   [[0,0,0,0],[1,1,1,1],[2,2,2,2]]
])
print("x shape: ",(x.shape))
print("swap(x,0,1) shape: ",(np.swapaxes(x,0,1).shape))
print("np.swapaxes(x,0,1): ", np.swapaxes(x,0,1))
print("swap(x,1,2) shape: ",(np.swapaxes(x,1,2).shape))
print("np.swapaxes(x,1,2): ", np.swapaxes(x,1,2))
print("swap(x,0,2) shape: ",(np.swapaxes(x,0,2).shape))
print("np.swapaxes(x,0,2): ", np.swapaxes(x,0,2))
```


``` 
결과: 
>>>
>>> x shape:  (2, 3, 4)
>>> swap(x,0,1) shape:  (3, 2, 4)
>>> np.swapaxes(x,0,1):  [[[0 0 0 0]
>>> [0 0 0 0]]
>>> 
>>> [[1 1 1 1]
>>> [1 1 1 1]]
>>> 
>>> [[2 2 2 2]
>>> [2 2 2 2]]]
>>> swap(x,1,2) shape:  (2, 4, 3)
>>> np.swapaxes(x,1,2):  [[[0 1 2]
>>> [0 1 2]
>>> [0 1 2]
>>> [0 1 2]]
>>> 
>>> [[0 1 2]
>>> [0 1 2]
>>> [0 1 2]
>>> [0 1 2]]]
>>> swap(x,0,2) shape:  (4, 3, 2)
>>> np.swapaxes(x,0,2):  [[[0 0]
>>> [1 1]
>>> [2 2]]
>>> 
>>> [[0 0]
>>> [1 1]
>>> [2 2]]
>>> 
>>> [[0 0]
>>> [1 1]
>>> [2 2]]
>>> 
>>> [[0 0]
>>> [1 1]
>>> [2 2]]]
```


# References
 [MINDSCALE, 2022.02.21, 부트스트랩](https://www.mindscale.kr/course/%ED%86%B5%EA%B3%84/%EB%B6%80%ED%8A%B8%EC%8A%A4%ED%8A%B8%EB%9E%A9)
