if __name__ == "__main__":
   # Imports
   from sklearn.model_selection import train_test_split
   from sklearn import datasets
   from NaiveBayes import NaiveBayes
   import numpy as np
   import matplotlib.pyplot as plt

   def accuracy(y_true, y_pred):
      accuracy = np.sum(y_true == y_pred) / len(y_true)
      return accuracy

   X, y = datasets.make_classification(
      n_samples=1000, n_features=10, n_classes=2, random_state=123
   )
   X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=123
   )

   # 시각화 함수.
   def visualize(X, y):
      print("X.shape: ", X.shape)
      print("X.head: ")
      print(X[:5][:])

      fig = plt.figure(figsize= (8,6))
      
      plt.scatter(
         X[:,0],
         X[:,1],
         c=y
         )
      plt.show()
      return 0

      
   def train_test(X_train, X_test, y_train, y_test):
      nb = NaiveBayes()
      nb.fit(X_train, y_train)
      predictions = nb.predict(X_test)

      print("Naive Bayes classification accuracy", accuracy(y_test, predictions))
      return 0
   
   # visualize(X, y)
   train_test(X_train, X_test, y_train, y_test)