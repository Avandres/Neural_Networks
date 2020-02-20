import numpy as np

class Perceptron():

    def __init__(self, n_ep=10, speed = 0.1):
        self.n_ep = n_ep
        self.speed_of_learning = speed

    def fit(self, X, y):
        print("Начато обучение...")
        self.w_array = np.zeros(X.shape[1] + 1)

        for epoha in range(self.n_ep):
            for i in range(X.shape[0]):
                delta = self.speed_of_learning * (y[i] - self.predict(X[i]))
                for j in range(0, len(self.w_array)-1):
                    self.w_array[j] += delta * X[i][j]
                self.w_array[len(self.w_array)-1] += delta

    def predict(self, x):
        x = np.append(x, 1)
        return np.where(sum(self.w_array*x) >= 0.0, 1, 0)




percept = Perceptron(n_ep=100, speed=0.01)


X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 1, 1, 1])

percept.fit(X, y)
print(percept.predict(X))

'''while True:
    x1 = int(input("Введите первое число\n"))
    x2 = int(input("Введите второе число\n"))
    print(x1, "or", x2, "=", percept.predict([x1, x2]), "\n")

from sklearn.linear_model import LinearRegression
import numpy as np
newLinearReg = LinearRegression()
X = np.array([[245, 38, 18, 62],
              [250, 40, 18, 85],
              [260, 40, 18, 85],
              [250, 40, 18, 85],
              [250, 40, 18, 85],
              [250, 40, 18, 85],
              [250, 40, 18, 85],
              [250, 40, 18, 85],
              [250, 40, 18, 85],
              [250, 40, 18, 85],
              [260, 40, 18, 85],
              [250, 40, 18, 85],
              [250, 40, 18, 85],
              [260, 40, 18, 85],
              [260, 40, 18, 85],
              [260, 40, 18, 85],
              [260, 40, 18, 85],
              [260, 40, 18, 85],
              [260, 40, 18, 85],
              [260, 40, 18, 85]])

y = np.array([380, 480, 480, 480, 440, 440, 420, 420, 420, 390, 390, 390, 450, 450, 440, 440, 450, 450, 450, 450])

newLinearReg.fit(X, y)
print(newLinearReg.predict([[245, 38, 18, 62]])[0])'''
