from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

iris = load_iris()
print(iris.feature_names)
print(iris.target_names)
test_indices = [0, 50, 100]

train_data = np.delete(iris.data, test_indices, axis=0)
train_target = np.delete(iris.target, test_indices)

test_data = iris.data[test_indices]
test_target = iris.target[test_indices]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)
print(test_target)
print(clf.predict(test_data))
