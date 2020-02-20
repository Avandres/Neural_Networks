from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()

'''from sklearn.datasets import load_iris
iris_dataset = load_iris()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

'''

class FMas():

    def __init__(self, mas, i):
        self.FeatureMas = [["all", 0]]
        for j in range(0, len(mas)):
            add = 1
            for k in range(1, len(self.FeatureMas)):
                if mas[j][i] == self.FeatureMas[k][0]:
                    add = 0
                    self.FeatureMas[k][1] += 1
                    self.FeatureMas[0][1] += 1
            if add == 1:
                self.FeatureMas.append([mas[j][i], 1])
                self.FeatureMas[0][1] += 1




class BayesianClassifier():
    # Only for binary signs
    def __init__(self):
        self.classMas = []

    def MaxNum(self, mas):
        max = [0, mas[0]]
        for i in range(0, len(mas)):
            if mas[i] > max[1]:
                max[0] = i
                max[1] = mas[i]
        return max

    def learn(self, X_train, y_train):
        for i in range(0, len(y_train)):
            add = 1
            for j in range(0, len(self.classMas)):
                if y_train[i] == self.classMas[j][0]:
                    add = 0
            if add == 1:
                mas = []
                self.classMas.append([y_train[i]])
                for k in range(i, len(y_train)):
                    if y_train[i] == y_train[k]:
                        mas.append(X_train[k])
                for k in range(0, len(X_train[i])):
                    newMas = FMas(mas, k)
                    self.classMas[len(self.classMas)-1].append(newMas)


    def predict(self, X_new):
        allClassesProb = []
        for j in range(0, len(self.classMas)):
            probability = 0
            for i in range(0, len(X_new)):
                for k in range(1, len(self.classMas[j][i+1].FeatureMas)):
                    if X_new[i] == self.classMas[j][i+1].FeatureMas[k][0]:
                        probability += self.classMas[j][i+1].FeatureMas[k][1]/self.classMas[j][i+1].FeatureMas[0][1]
            allClassesProb.append(probability)

        predClass = self.MaxNum(allClassesProb)
        return self.classMas[predClass[0]][0]


X_train = [[1, 0, 1, 0, 0, 0, 0],
           [0, 1, 0, 1, 0, 0, 0],
           [0, 1, 0, 0, 0, 1, 1],
           [1, 0, 0, 0, 1, 0, 0],
           [1, 0, 0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0, 1, 1],
           [1, 0, 1, 0, 0, 0, 0],
           [0, 1, 0, 1, 0, 0, 1],
           [0, 1, 1, 0, 0, 0, 1]]
y_train = ["Птица", "Рыба", "Животное", "Насекомое", "Насекомое", "Животное", "Птица", "Рыба", "Птица"]
X_test = [[1, 0, 1, 0, 0, 0, 0],
          [1, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 1],
          [0, 0, 1, 0, 0, 0, 0]]
y_test = ["Птица", "Насекомое", "Животное", "Птица"]

newBayes = BayesianClassifier()
newBayes.learn(X_train, y_train)

divide = [0, 0]
for i in range(0, len(X_test)):
    if newBayes.predict(X_test[i]) == y_test[i]:
        divide[0] += 1
    else:
        divide[1] += 1

print(divide[0]/len(y_test))
