from sklearn import datasets
from sklearn.cross_validation import train_test_split as tp
from sklearn.tree import DecisionTreeClassifier


def accuracyScore(a, b):
    sum = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            sum += 1
    return(sum/len(a))


iris = datasets.load_iris()

x = iris.data
y = iris.target

xTrain, xTest, yTrain, yTest = tp(x, y, test_size=0.5)

dtClassifier = DecisionTreeClassifier()
dtClassifier.fit(xTrain, yTrain)

prediction = dtClassifier.predict(xTest)

print("Accuracy of DT Classifer: {}".format(accuracyScore(prediction, yTest)))
