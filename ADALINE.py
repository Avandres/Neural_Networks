import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PerceptronADALINE():

    def __init__(self, n_ep=10, speed = 0.1):
        self.n_ep = n_ep
        self.speed_of_learning = speed

    def fit(self, X, y):
        print("Начато обучение...")
        for i in range(len(X)):
            X[i].insert(0, 1.0)
        X = np.array(X)
        self.w_array = np.zeros(X.shape[1])

        for epoha in range(self.n_ep):
            for i in range(len(self.w_array)):
                for j in range(X.shape[1]):
                    s = 0
                    for k in range(X.shape[0]):
                        s += y[k] - np.dot(self.w_array, X[k]) * X[k][j]
                    self.w_array[j] += self.speed_of_learning * s

    def predict(self, x):
        return np.where((self.w_array[0] + sum(self.w_array[1:]*x)) >= 0.0, 1, 0)




percept = PerceptronADALINE(n_ep=1000, speed=0.1)

X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

y = np.array([0., 1., 1., 1.])



percept.fit(X, y)
print(percept.w_array)




while True:
    x1 = int(input("Введите первое число\n"))
    x2 = int(input("Введите второе число\n"))
    print(x1, "or", x2, "=", percept.predict([x1, x2]), "\n")
