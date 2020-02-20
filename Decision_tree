import math
import numpy as np
import copy


class tree_classifier():

    def __init__(self):
        self.KoeffMas = [""]
        self.answer = [""]
        self.restructions = [""]

    def Entropy(self, y_mas):
        entropy = 0
        allClasses = []
        for i in range(0, len(y_mas)):
            add = 1
            for j in range(0, len(allClasses)):
                if y_mas[i] == allClasses[j][0]:
                    add = 0
                    allClasses[j][1] += 1
                    break
            if add == 1:
                allClasses.append([y_mas[i], 1])

        for i in range(0, len(allClasses)):
            p = allClasses[i][1] / len(y_mas)
            entropy += p * math.log2(p)
        entropy *= (-1)
        return entropy

    def feature_and_limitation(self, X_train, y_train):
        baseEntropy = self.Entropy(y_train)
        min_Entropy_in_Row = []
        min_Entropy_in_Cell = []

        for i in range(0, len(X_train[0])):

            mas = []
            for j in range(0, len(X_train)):
                mas.append(X_train[j][i])
            if self.is_only_one(mas):
                continue

            for j in range(0, len(X_train)):
                y1 = []
                y2 = []

                a = X_train[j][i]
                for k in range(0, len(y_train) - 1):
                    if X_train[k][i] > a:
                        y1.append(copy.copy(y_train[k]))
                    else:
                        y2.append(copy.copy(y_train[k]))

                newEntropy = (self.Entropy(y1) + self.Entropy(y2)) / 2
                if y1 == []:
                    continue

                if not min_Entropy_in_Row:
                    min_Entropy_in_Row = [a, newEntropy]
                else:
                    if newEntropy < min_Entropy_in_Row[1]:
                        min_Entropy_in_Row = [a, newEntropy]

            if not min_Entropy_in_Row:
                return "no more!.."
            if not min_Entropy_in_Cell:
                min_Entropy_in_Cell = [i, min_Entropy_in_Row[0], min_Entropy_in_Row[1]]
                min_Entropy_in_Row = []
            else:
                if min_Entropy_in_Row[1] < min_Entropy_in_Cell[2]:
                    min_Entropy_in_Cell = [i, min_Entropy_in_Row[0], min_Entropy_in_Row[1]]
                    min_Entropy_in_Row = []
        if min_Entropy_in_Cell != []:
            return min_Entropy_in_Cell
        else:
            return "no more!.."

    def is_only_one(self, mas):
        answ = 1
        value = mas[0]
        for i in range(0, len(mas)):
            if mas[i] != value:
                answ = 0
                break
        return answ

    def most_frequent(self, mas):
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

    def answ(self, y_train, numb_of_knot):
        while len(self.KoeffMas) < numb_of_knot + 1:
            self.KoeffMas.append("")
            self.restructions.append("")
            self.answer.append("")
        self.KoeffMas[numb_of_knot] = "answer"
        self.restructions[numb_of_knot] = "answer"
        if len(y_train) > 0:
            self.answer[numb_of_knot] = self.most_frequent(y_train)

    def learn(self, X_train, y_train, k, n=1, numb_of_knot=1):
        newMasX = []
        newMasY = []

        if (k != n) and (len(X_train) > 0) \
                and (self.is_only_one(y_train) != 1) and (len(y_train) > 0):
            newSign = self.feature_and_limitation(X_train, y_train)
            if newSign == "no more!..":
                self.answ(y_train, numb_of_knot)
            else:
                while len(self.KoeffMas) < numb_of_knot + 1:
                    self.KoeffMas.append("")
                    self.restructions.append("")
                self.KoeffMas[numb_of_knot] = newSign[0]
                self.restructions[numb_of_knot] = newSign[1]
                while len(self.answer) < numb_of_knot + 1:
                    self.answer.append("")
                self.answer[numb_of_knot] = ""

                for i in range(0, len(X_train)):
                    if X_train[i][newSign[0]] > newSign[1]:
                        newMasX.append(copy.copy(X_train[i]))
                        newMasY.append(copy.copy(y_train[i]))
                if newMasY != []:
                    self.learn(newMasX, newMasY, k, n + 1, numb_of_knot + numb_of_knot + 1)
                else:
                    self.answ(y_train, numb_of_knot)

                newMasX = []
                newMasY = []

                for i in range(0, len(X_train)):
                    if X_train[i][newSign[0]] <= newSign[1]:
                        newMasX.append(copy.copy(X_train[i]))
                        newMasY.append(copy.copy(y_train[i]))
                if newMasY != []:
                    self.learn(newMasX, newMasY, k, n + 1, numb_of_knot + numb_of_knot)
                else:
                    self.answ(y_train, numb_of_knot)

        else:
            self.answ(y_train, numb_of_knot)

    def predict(self, X_new, numb_of_knot=1):
        if self.KoeffMas[numb_of_knot] == "answer":
            return self.answer[numb_of_knot]
        elif X_new[self.KoeffMas[numb_of_knot]] > self.restructions[numb_of_knot]:
            return self.predict(X_new, numb_of_knot + numb_of_knot + 1)
        elif X_new[self.KoeffMas[numb_of_knot]] <= self.restructions[numb_of_knot]:
            return self.predict(X_new, numb_of_knot + numb_of_knot)


'''import random
import copy
random.seed()
import numpy as np

from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    cancer_dataset['data'], cancer_dataset['target'], random_state=0)


new_tree = tree_classifier()
new_tree.learn(X_train, y_train, 20)
divide = [0, 0]
for i in range(0, len(X_test)):
    if new_tree.predict(X_test[i]) == y_test[i]:
        divide[0] += 1
    else:
        divide[1] += 1

print(divide[0] / len(y_test))'''
