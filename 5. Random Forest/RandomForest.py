import numpy as np
from DecisionTree import DecisionTree

class RandomForest():
   def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_features=None):
      self.n_trees = n_trees
      self.min_samples_split = min_samples_split
      self.max_depth = max_depth
      self.n_features = n_features
      self.trees = []

   def fit(self, X, y):
      """ loop 설명
         n_trees 개만큼 tree를 생성하는 loop:
            각 loop 마다 X_sampled, y_sampled 를 생성,
            이 데이터로 tree를 구성하여 self.tree instance를 array 에 append 해준다.
         
         최종.
         >>> self.tree = [DecisionTree1, DecisionTree1, ..., DecisionTree_{n_tree}] 가 저장된다. 
      """
      for _ in range(self.n_trees):
         tree = DecisionTree(self.min_samples_split, self.max_depth, self.n_features)
         """ Boostrapping(같은 크기로 무작위 복원추출)
            X와 y를 tree에 그대로 넣는 게 아니라. X_samples, y_samples 를 생성하고
            그걸 decisiontree 모델 생성에 사용한다.
         """
         X_sampled, y_sampled = self._boostrap_samples(X, y)
         tree.fit(X_sampled, y_sampled)
         self.trees.append(tree)


   def _boostrap_samples(self, X, y):
      """ fit > X_sample, y_sample // 부트스트래핑
         크기는 모집단과 동일(X.shape[0]), np.random.choice() 함수 사용하여 idxs를 생성한다.
         그리고 X[idxs], y[idxs]를 반환하면 깔끔하게 무작위복원추출을 할 수 있음.
      """
      n_samples = X.shape[0]
      idxs = np.random.choice(n_samples, n_samples, replace=True)
      return X[idxs], y[idxs]
   
   def predict(self, X):
      """ 설명
         self.tree의 요소에 접근해서 self.tree[idx].predict()를 n번 적용. \n 
         predictions = [pred_1, pred_2, ..., pred_{n_tree}] 에 저장 후
         predictions = [
                        [1,0,...,1],
                        [0,0,...,1],
                        ...,
                        [1,1,...,0]
                        ]
         앙상블을 적용 한다.
         
      """
      predictions = np.array([tree.predict(X) for tree in self.trees])
      tree_preds = np.swapaxes(predictions, 0, 1)
      predictions = np.array([DecisionTree()._most_common_label(pred) for pred in tree_preds])
      return predictions
   


