import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from KNN import KNN

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


## X 시각화.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

plt.figure()
plt.scatter(X[:,2],
            X[:,3], 
            c=y, 
            cmap=cmap, 
            edgecolor='k', 
            s=20)
plt.show()
"""

# knn 클래스 인스턴스 생성
clf = KNN(k=5)
# 데이터 클래스로 전달.
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print(predictions)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)