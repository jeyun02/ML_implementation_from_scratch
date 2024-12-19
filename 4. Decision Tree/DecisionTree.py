import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*, value=None):
        """
        이진트리구조에 필요한 변수들은 무엇이 있을까
        \n노드의 left 자식노드를 가리키는 포인터,
        \n노드의 right 자식노드를 가리키는 포인터,
        \n노드의 값 관련.( 입력값, 기준변수, 기준치, 부등호 기준)
        """
        # 함수설명 
        self.feature =feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


    def is_leaf_node(self):
        return self.value is not None



class DecisionTree():
    def __init__(self, min_samples_split=2, max_depth=100, impurity="entropy", n_features=None):
        """변수설정. 함수 인자에서 정의하고 초기값을 None 으로 하는 것과
        함수 인자에서는 정의 안했지만 self.변수 에서는None 으로 초기화 하는 것의 차이는 무엇일까.\n
        입력 받을 수 있냐 없냐 차이겠지?\n
        n_feature 이 None 이기 때문에 random으로 feature를 선정하지 않고 30개의 모든 feature 를 사용함."""
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.impurity = impurity
        self.n_features = n_features # 랜덤으로 뽑을 feature 개수( hyperparameter 가 되나)
        self.root = None
        
    def fit(self, X, y):
        """ fit 함수란 반복적으로 사용되면서 함수의 coefficient 를 변경하는 함수"""
        # tree의 n_features 은 데이터의 열 수를 넘으면 당연히 안되겠지. 
        # 일단 초기 값은 x의 열 수 로 놓은 다음에 점점 줄여나가는 건가?
        # n_feature 이 None 이기 때문에 random으로 feature를 선정하지 않고 30개의 모든 feature 를 사용함.
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        """ fit > self.root """
        n_samples, n_feats = X.shape # n_feats개 columns 중 랜덤으로 self.n_features개만큼 뽑겠다~
        n_labels = len(np.unique(y))
        
        # [o] 1. 전체적으로 재귀함수로서 활용될테니, 정지조건을 설정.
        """ 조건문 설명
            오버피팅을 막기 위해 더 이상 깊게 하지 못하게 max_depth 설정값보다 현재 depth가 크거나 같으면 멈춘다. 
            or 데이터의 y 범주가 1개 ( 다 같은 색깔의 점) 이면 멈춘다.
            or 트리를 split 할 수 있는 최소 샘플 수(2) 보다 샘플 수가 작을 때 멈춘다.
            이때 리턴은 최다 label 을 제출.
        """        
        if(depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # [o] 2. 최적의 split 을 찾는 함수( 통계적 과정)
        """ 설명
            X에서 n_feats 개 columns 중 랜덤으로 self.n_features개 뽑겠다.
            길이가 self.n_features인 ndarray가 반환된다.
        """
        feature_idxs = np.random.choice(n_feats, self.n_features, replace=False) 
        best_feature, best_thresh  = self._best_split(X, y, feature_idxs)
        
        # [o] 3. 최적의 split 으로 left tree와 right tree 를 재귀로 생성후 root 노드 반환
        """ 설명
            \n위 2. 에서 1순위 split feature 와 threshold 을 찾았음.
            \n이를 기준으로 좌 우 서브트리를 덛붙여야함. 
            \n이때 _grow_tree를 재귀함수로서 사용함으로써 서브트리에서 더 split이 가능한지 아닌지 1. 조건으로 판단 가능  
            \n최종 root Node 인스턴스를 반환.
        """
        left_idxs, right_idxs = self._split(X[:,best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs,:], y[left_idxs], depth+1) # recursion point 1
        right = self._grow_tree(X[right_idxs,:], y[right_idxs], depth+1) # recursion point 2
        return Node(best_feature, best_thresh, left, right)
    
    def _best_split(self, X, y, feat_idxs):
        """ fit > _grow_tree > 2. 최적의 split 찾기.
            각 column 에 대해 반복.
            e.g. 나이 column 에 대해 모든 값들이 threshold 후보가 됨.
            이 때 모든 값들을 구분선으로 했을 때 information gain 을 계산한다.
            그래서 정보량이 최대가 되는 기준을 찾는거지!"""
        best_gain = -1
        split_idx, split_threshold = None, None
        
        """ First Loop  
            랜덤으로 뽑은 모든 X feaure 들 각각에 대해 split이 가능한 모든 후보 array(thresholds)를 만든다. 
        """
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column) # 두 개로 쪼갤 수 있는 모든 기준점들을 전부 쪼개보고, 정보량을 계산한다!
        
            """ Second Loop
                한 feature의 모든 split point 후보들(thresholds)에 대해 Information gain을 계산 후 
                정보량이 최대인 feature와 threshold의 조합을 찾는다.
                e.g. 가령 (feat, thr, IG )의 조합이 
                (나이 , 33, 0.30)
                (키, 170, 0.29)
                (몸무게, 80, 0.27)
                (키, 179, 0.26)
                ...
                나오겠지?
                이중에 가장 좋은 조합은 idx = 나이 idx , thr = 33 이므로 이걸 최종 return 해준다.
            """
            for thr in thresholds:
                # thr 보다 작은 X_column 값의 idx 를 구한다.
                # 그 y[idx]값을 가지고 정보량을 계산한다.
                gain = self._information_gain(y, X_column, thr)

                if (gain > best_gain) :
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        
        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        """ fit > _grow_tree > 2._best_split"""
        
        # 1. parents entropy
        parents_impurity = self._impurity(y)

        # 2. children 
        left_idxs, right_idxs = self._split(X_column, threshold)
        if (len(left_idxs) == 0 or len(right_idxs) == 0):
            """ 설명:
                \n이 조건에 해당한다는 것은 threshold 가 정렬된 양 끝단에 있다는 것이니. 
                0/n + 1 * log(1)이 발생해 그 threshold에 대한 information 값은 0이 나온다.
                \n (즉, 계산해도되지만, 오류 가능성과 계산 축소를 위해 조건문으로 0을 걸러준다.
            """
            return 0
        
        # 3. weighted average of children E
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._impurity(y[left_idxs]), self._impurity(y[right_idxs])
        children_impurity = (n_l/n)*e_l + (n_r/n)*e_r
        
        # 4. IG = parents E - weighted average of children E
        informain_gain = parents_impurity - children_impurity
        return informain_gain
    
    def _impurity(self, y):
        """ fit > _grow_tree > 2._best_split > _information_gain > 1,2,3,4
        최하단 information(=entropy) 값을 구한다.
        using np.bincount(y):
        e.g. 
        >>> y = [1,2,3,1,2] 
        >>> np.bincount(y)
        >>> [0, 2, 2, 1] 0은 0번, 1은 2번, 2는 2번, 3은 1번 등장했다는 뜻 
            -> histogram 만들기 용이. len(y) 로 나누면 p(label) array 를 만들 수 있음.
        """

        #  value = counter.most_common(1)[0][0]에서 
        # index out of range 에러 발생하여
        # y =[] 인 경우 0 리턴해준다.
        # 2024.12.19. 
        if len(y) == 0: return 0

        hist = np.bincount(y)
        p_labels = hist / len(y)


        if self.impurity == "entropy": # default.
            return -np.sum([p * np.log(p) for p in p_labels if (p > 0)]) # log 의 밑 값은 상관 없음. 최적의 threshold 를 찾기만 하면 되니까.
        elif self.impurity == "gini":
            return 1 - np.sum([p ** 2 for p in p_labels if (p > 0)])
        elif self.impurity == "misclassification":
            return 1 - np.max(p_labels) if len(p_labels) > 0 else 0
        else:
            print("\"impurity = ",self.impurity,"\"is not in the list of impurity selection.")
            print("The impurity is caculated by default setting(entropy).")
            print("If you want other impurity, choose one of [\"entropy\", \"gini\", \"misclassification\"].")



    def _split(self, X_column, split_threshold):
        """ fit > _grow_tree > 2._best_split > _information_gain > 3.children , \n fit > _grow_tree > 3.
            \narray를 split_threshold 기준 큰 idx 는 right_dixs 로, 
            나머지는 left_idxss의 두개의 array로 쪼개주는 단순한 함수.
        """
        """ np.argwhere(조건식) 설명
            >>> x =[10,20,30,40,50]
            >>> np.argwhere(x<40)
            >>> array([[0],[1],[2]])
        """
        left_idxs = np.argwhere(X_column<=split_threshold).flatten()
        right_idxs = np.argwhere(X_column>split_threshold).flatten()
        return left_idxs, right_idxs
    
    def _most_common_label(self, y):
        """ fit > _grow_tree > 1. leaf_value
            정지 조건 해당 시 leaf_node 의 value 로써 사용.   
            \ny(array)에서 가장 많은 labels 종류를 반환한다.
            \nCounter().most_common(상위 몇개)은 [[(요소,요소개수)]] 형식의 리스트 내 튜플 형식을 반환.   
            \n따라서 counter.most_common(1)[0][0] 란, 요소 개수 순으로 1등 요소에 대해,   
            0번째 리스트요소에 들어가고, 0번째 튜플요소에 들어가면 가장 많이 등장한 y 요소를 알 수 있다.
        """

        # y = [] 인 경우 에러 발생. -> None 리턴.

        if len(y) == 0 : return None

        counter = Counter(y)
        most_common = counter.most_common(1)
        return most_common[0][0] if most_common else None # most_common = [] 인 경우도 에러가 나므로 return None 해준다.

    def predict(self, X):
        """ 함수 설명
            x: X의 한 행
            e.g. x = [ 33(살), 175(cm), 70(kg)] 
            우리가 최적화시킨 Tree의 root 노드부터 적용 시켜
            y 값이 나오는 것을 저장한다.
        """
        rs = np.array([self._traverse_tree(x, self.root) for x in X] )
        return rs
    
    def _traverse_tree(self, x, node):
        """ predict > . // 함수 설명
            \n이미 self.root 는 fit 된 Tree 노드의 root임.
            \n일단 is_leaf 가 아닐때까지만 loop를 돈다. leaf 면 self.root.value 반환!
            \n    x라는 (1 x n_feats )크기의 array 에서 self.root.feature 에 해당하는 값을 뽑고
            \n    self.root.threshold 와 X를 비교해서 
            \n    작으면 self.root.left 에 대해 _traverse 재귀
            \n    크면 self.root.right 에 대해 _traverse 재귀
        """
        if node is None: return None # None 예외처리
        if node.is_leaf_node():
            return node.value
        if node.threshold is None :
            print(f"Warning: Node {node} has None threshold")
            return node.value # None 예외처리
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)