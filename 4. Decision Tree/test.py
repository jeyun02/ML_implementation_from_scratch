import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTree  # Import your DecisionTree class

def accuracy(y_test, y_pred):
    accuracy = np.sum(y_test == y_pred) / len(y_test)
    return accuracy


data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)
impurities = ["entropy","gini","misclassification"]
impurity_selection = impurities[2] # select in impurities.
clf = DecisionTree(max_depth=10, impurity=impurity_selection)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

acc = accuracy(y_test, predictions)
print ("Accuracy is ", acc, "with ", impurity_selection, "impurity function!")