from TreeClassifier import tree_classifier as tree
import random
import copy


random.seed()

from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    cancer_dataset['data'], cancer_dataset['target'], random_state=0)


class random_forest_classifier():


    def __init__(self, N):

        self.voting = []
        self.forest = []
        for i in range(0, N):
            new_tree = tree()
            self.forest.append(new_tree)

    def shiffle(self, X_train, y_train):
        new_X_train = []
        new_y_train = []
        for i in range(0, len(X_train)):
            rand = random.randint(0, len(X_train) - 1)
            new_X_train.append([])
            new_y_train.append([])
            new_X_train[i] = copy.copy(X_train[rand])
            new_y_train[i] = copy.copy(y_train[rand])
        return new_X_train, new_y_train

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

    def learn_forest(self, X_train, y_train, tree_size):

        for i in range(0, len(self.forest)):
            train = self.shiffle(X_train, y_train)
            self.forest[i].learn(train[0], train[1], tree_size)
            print(i)

    def predict_forest(self, X_new):

        self.voting.clear()
        for i in range(0, len(self.forest)):
            self.voting.append(self.forest[i].predict(X_new))
        return self.most_frequent(self.voting)


new_forest = random_forest_classifier(3)
new_forest.learn_forest(X_train, y_train, 20)


divide = [0, 0]
for i in range(0, len(X_test)):
    if new_forest.predict_forest(X_test[i]) == y_test[i]:
        divide[0] += 1
    else:
        divide[1] += 1

print(divide[0]/len(y_test))


