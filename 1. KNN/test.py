import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


## X 시각화.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF']) # red green blue 의 color hex code 임.

plt.figure()
plt.scatter(X[:,2],
            X[:,3], 
            c=y, 
            cmap=cmap, 
            edgecolors='k', 
            s=20)
plt.show()
"""

from KNN import KNN

clf = KNN(k=5) # class instance 생성
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)   

print(predictions)




"""import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

plt.figure()
plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()


clf = KNN(k=5)
# 데이터 클래스로 전달.
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print(predictions)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)