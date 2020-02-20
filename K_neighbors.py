import math
from sklearn.datasets import load_iris
iris_dataset = load_iris()
from sklearn.model_selection import train_test_split


def Xrange(X1, X2):
    distance = 0
    for i in range(0, len(X1)):
        distance += pow((X1[i] - X2[i]), 2)
    distance = math.sqrt(distance)
    return distance


def MinNum(mas):
    min = [0, mas[0]]
    for i in range(0, len(mas)):
        if mas[i] < min[1]:
            min[0] = i
            min[1] = mas[i]
    return min

def MaxNum(mas):
    max = [0, mas[0]]
    for i in range(0, len(mas)):
        if mas[i] > max[1]:
            max[0] = i
            max[1] = mas[i]
    return max

def minMaxNormalize(mas):
    normalMas = []
    for i in range(0, len(mas[0])):
        for j in range(0, len(mas)):
            normalMas.append(mas[j][i])
        for j in range(0, len(mas)):
            minNorm = MinNum(normalMas)
            maxNorm = MaxNum(normalMas)
            mas[j][i] = (mas[j][i] - minNorm[1])/(maxNorm[1] - minNorm[1])
        normalMas = []
    return mas


def most_frequent(mas):
    clMas = []
    for i in range(0, len(mas)):
        add = 1
        for j in range(0, len(clMas)):
            if mas[i] == clMas[j][0]:
                add = 0
                clMas[j][1] += 1
                break
        if add == 1:
            clMas.append([mas[i], 1])

    max = clMas[0]
    for i in range(0, len(clMas)):
        if max[1] < clMas[i][1]:
            max = clMas[i]

    return max[0]




def main(X, X_new, y, k):
    distMas = []
    minDistancePoint = []
    for i in range(0, len(X)):
        dist = Xrange(X[i], X_new)
        distMas.append(dist)

    for i in range(0, k):
        kek = MinNum(distMas)
        minDistancePoint.append(y[kek[0]])
        del(distMas[kek[0]])


    cl = most_frequent(minDistancePoint)

    return cl




iris_dataset['data'] = minMaxNormalize(iris_dataset['data'])
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)



Xkek = X_test[0]
cl = main(X_train, Xkek, y_train, 5)
print("Предсказан класс ", cl)

print(X_test[0])
print(y_test[0], "\n")

for i in range(0, len(X_test)):
    if main(X_train, X_test[i], y_train, 5) == y_test[i]:
        print("+")
    else:
        print("-")
