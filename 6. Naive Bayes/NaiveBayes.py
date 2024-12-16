import numpy as np

class NaiveBayes:

   def fit(self, X, y):
      n_samples, n_features = X.shape # (200, 4) 중 120 행이 y==yes 라 해보자
      self._labels = np.unique(y) # y = [0,1]
      n_labels = len(self._labels) # 2개 : {1 or 0}

      # calculate mean, var, and prior for each class
      # e.g. self.mean
      # >>>
      # [[mean_x1_yes, mean_x2_yes, mean_x3_yes, mean_x4_yes],
      #  [mean_x1_no , mean_x2_no , mean_x3_no , mean_x4_no ]]
      self._mean = np.zeros((n_labels, n_features), dtype=np.float64)
      self._var = np.zeros((n_labels, n_features), dtype=np.float64)
      self._priors = np.zeros(n_labels, dtype=np.float64)
      
      """ enumerate 설명
         e.g.
         >>> self._labels
         >>> ["yes", "no"]
         >>> enumerate(self._labels)
         >>> [(0, "yes"), (1, "no")]
         즉, 아래 코드에서 idx 가 0,1일 때 각각 동시에 label 은 "yes","no"인 것임.
      """
      for idx, label in enumerate(self._labels):
         """ e.g.
            idx : 0 , label : "yes"
            >>> X_label 
            >>> [
            >>> [x1_yes_1, x2_yes_1, x3_yes_1, x4_yes_1],
            >>> [x1_yes_2, x2_yes_1, x3_yes_1, x4_yes_1],
            >>> ...,
            >>> [x1_yes_120, x2_yes_120, x3_yes_120, x4_yes_120]]
            >>> 
            >>> np.mean(X_label) # 가장 깊은 곳에서 mean 을 계산해준다..
            >>> [mean_1, mean_2, ..., mean_120]
         """
         X_label = X[y == label] 
         self._mean[idx, :] = np.mean(X_label, axis=0)
         self._var[idx, :] = np.var(X_label, axis=0)
         self._priors[idx] = X_label.shape[0] / float(n_samples) # 120 / 200.0
         """# ? self._priors[idx] 설명
            >>> X_label.shape[0]
            >>> ( 120, 1)[0]
            >>> 120
            >>>
            >>> n_samples
            >>> 200
            즉, p(y=="yes") = 120/200, p(y="no") = 80/200 을 계산한거네
         """

         pass
   def predict(self, X):
      y_pred = [self._predict(x) for x in X]
      return np.array(y_pred)

   def _predict(self, x):
      posteriors = []

      for idx, label in enumerate(self._labels):
         prior = np.log(self._priors[idx])
         posterior = np.sum(np.log(self._pdf(idx, x)))
         posterior += prior
         posteriors.append(posterior)
      
      return self._labels[np.argmax(posteriors)]

   def _pdf(self, idx, x):
      mean = self._mean[idx]
      var =  self._var[idx]
      numerator = np.exp(-((x-mean) ** 2) / (2*var))
      denominator = np.sqrt(2 * np.pi * var)
      return numerator / denominator
