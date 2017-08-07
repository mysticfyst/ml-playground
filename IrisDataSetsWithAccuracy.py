from sklearn import datasets
from sklearn.cross_validation import train_test_split as tp

iris = datasets.load_iris()

x = iris.data
y = iris.target

xTrain, xTest, yTrain, yTest = tp(x, y, test_size=0.5)
print(xTrain)
