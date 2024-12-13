import numpy as np
from collections import Counter

class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None,*, value=None):
        """
        이진트리구조에 필요한 변수들은 무엇이 있을까
        노드의 left 자식노드를 가리키는 포인터,
        노드의 right 자식노드를 가리키는 포인터,
        노드의 값 관련.( 입력값, 기준변수, 기준치, 부등호 기준)
        """
        # 함수설명 
        self.feature =feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = None


    def is_leaf_node(self):
        return self.value is not None



class DecisionTree():
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        """변수설정. 함수 인자에서 정의하고 초기값을 None 으로 하는 것과
        함수 인자에서는 정의 안했지만 self.변수 에서는None 으로 초기화 하는 것의 차이는 무엇일까.
        입력 받을 수 있냐 없냐 차이겠지?
        n_feature 의 초기값을 None으로 인자로 설정한 것은"""
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features # 랜덤으로 뽑을 feature 개수( hyperparameter 가 되나)
        self.root = None
        
    def fit(self, X, y):
        """ fit 함수란 반복적으로 사용되면서 함수의 coefficient 를 변경하는 함수"""
        # tree의 n_features 은 데이터의 열 수를 넘으면 당연히 안되겠지. 
        # 일단 초기 값은 x의 열 수 로 놓은 다음에 점점 줄여나가는 거 같네.
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape # n_feats개 columns 중 랜덤으로 self.n_features개만큼 뽑겠다~
        n_labels = len(np.unique(y))
        
        # 1. 전체적으로 재귀함수로서 활용될테니, 정지조건을 설정.
        """
        오버피팅을 막기 위해 더 이상 깊게 하지 못하게 max_depth 설정값보다 현재 depth가 크거나 같으면 멈춘다. 
        or 데이터의 y 범주가 1개 ( 다 같은 색깔의 점) 이면 멈춘다.
        or 트리를 split 할 수 있는 최소 샘플 수(2) 보다 샘플 수가 작을 때 멈춘다.
        
        이때 리턴은 최다 label 을 제출.
        """
        if(depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # 2. 최적의 split 을 찾는 함수( 통계적 과정)
        feature_idxs = np.random.choice(n_feats, self.n_features, replace=False) 
        # X에서 n_feats 개 columns 중 랜덤으로 self.n_features개 뽑겠다.
        # 길이가 self.n_features인 ndarray가 반환된다.
        best_threshold, best_feature = self._best_split(X, y, feature_idxs)
        
        # 3.서브트리를 생성하고 그 root에 재귀함수를 걸어서 뻗어나가도록.

    def _best_split(self, X, y, feat_idxs):
        """ fit- 2. 최적의 split 찾기.
            각 column 에 대해 반복.
            e.g. 나이 column 에 대해 모든 값들이 threshold 후보가 됨.
            이 때 모든 값들을 구분선으로 했을 때 information gain 을 계산한다.
            그래서 정보량이 최대가 되는 기준을 찾는거지!"""
        best_gain = -1
        split_idx, split_threshold = None, None
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column) # 두 개로 쪼갤 수 있는 모든 기준점들을 전부 쪼개보고, 정보량을 계산한다!
        
            for thr in thresholds:
                # thr 보다 작은 X_column 값의 idx 를 구한다.
                # 그 y[idx]값을 가지고 정보량을 계산한다.
                gain = 1
                """
                CHECKLIST
                [ ]: 할일 1 
                [ ]: 할일 2 
                """
                if (gain > best_gain) :
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        
        return split_idx, split_threshold



    def _most_common_label(self, y):
        """ fit-1. 정지 조건 시 leaf_node 의 value 로써 사용.
            y(array)에서 가장 많은 labels 개수(int)를 반환한다
            Counter().most_common(상위 몇개)은 [[(요소,요소개수)]] 형식의 리스트 내 튜플 형식을 반환.
            따라서 counter.most_common(1)[0][0] 란, 요소 개수 순으로 1등 요소에 대해,
            0번째 리스트요소에 들어가고, 0번째 튜플요소에 들어가면 가장 많은 요소의 개수를 알 수 있다."""
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    
    def predict(self, ):
        """
        TODO 오 됐다.
        왜 안되냐"""
        pass