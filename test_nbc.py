from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm

#loading and splitting the dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Naive bayes
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

#SVN
clf = svm.SVC()
y_pred2 = clf.fit(X_train, y_train).predict(X_test)

print("SVM")
print("total %d mislabeled points : %d"
      % (X_test.shape[0], (y_test != y_pred2).sum()))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred2))

print("Naive Bayes")
print("total %d mislabeled points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))