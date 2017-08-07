from sklearn import datasets
from sklearn.cross_validation import train_test_split as tp
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance


def euc(a, b):
    return distance.euclidean(a, b)


class KNN():

    def fit(self, xTrain, yTrain):
        self.xTrain = xTrain
        self.yTrain = yTrain

    def predict(self, xTest):
        predictions = []
        for row in xTest:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        shortestD = euc(row, self.xTrain[0])
        index = 0
        for i in range(1, len(self.xTrain)):
            dist = euc(row, self.xTrain[i])
            if dist < shortestD:
                index = i
                shortestD = dist
        return self.yTrain[index]


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

stockKNN = KNeighborsClassifier()
stockKNN.fit(xTrain, yTrain)

prediction1 = stockKNN.predict(xTest)

customKNN = KNN()
customKNN.fit(xTrain, yTrain)

prediction2 = customKNN.predict(xTest)

print("Accuracy of stock KNN Classifer: {}".format(accuracyScore(prediction1,
                                                                 yTest)))
print("Accuracy of custom KNN Classifer: {}".format(accuracyScore(prediction2,
                                                                  yTest)))
