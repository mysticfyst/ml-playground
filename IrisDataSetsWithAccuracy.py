from sklearn import datasets
from sklearn.cross_validation import train_test_split as tp
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


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

prediction1 = dtClassifier.predict(xTest)

knnClassifier = KNeighborsClassifier()
knnClassifier.fit(xTrain, yTrain)

prediction2 = knnClassifier.predict(xTest)

score1 = accuracyScore(prediction1, yTest)
score2 = accuracyScore(prediction2, yTest)

print(score1, score2, sep='  ')

if score1 > score2:
    print("Decision Tree performs better")
elif score2 > score1:
    print("KNN performs better")
else:
    print("Both the classifiers perform the same")
