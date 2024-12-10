
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

import matplotlib.pyplot as plt

# data loading
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


# Making a classifier
clf = LogisticRegression(lr =1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 지표 평가.. check the accuracy of y_pred and y_test
def accuracy(a, b):
   # 맞은개수 / 전체개수
   return np.sum(a==b) / len(b)

print(accuracy(y_pred, y_test))