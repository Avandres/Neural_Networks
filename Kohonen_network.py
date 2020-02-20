import numpy as np
import random
from math import sqrt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

class KohonenNetwork():

    def __init__(self, number_of_clusters, learning_speed, number_of_epochs):
        self.n_clusters = number_of_clusters
        self.ls = learning_speed
        self.number_of_epochs = number_of_epochs

    def fit(self, data):
        self.w = np.random.sample((self.n_clusters, data.shape[1]))
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                self.w[i, j] = random.triangular(low=0.5-(1/sqrt(data.shape[1])), high=0.5+(1/sqrt(data.shape[1])))
        for e in range(self.number_of_epochs):
            for ind in range(data.shape[0]):
                vec = random.choice(data)
                predicted_cluster = self.predict(np.array([vec]))
                self.w[predicted_cluster[0], :] = self.w[predicted_cluster[0], :] + \
                                                  self.ls*(data[ind, :] - self.w[predicted_cluster[0], :])

    def self_organization(self, data, Rmax):
        self.w = np.array([random.choice(data)])
        for e in range(self.number_of_epochs):
            del_neurons_mas = []
            for i in range(self.w.shape[0]):
                del_neurons_mas.append(True)
            for ind in range(data.shape[0]):
                vec = random.choice(data)
                new_neuron = True
                for j in range(self.w.shape[0]):
                    if self.find_R(vec, self.w[j, :]) <= Rmax:
                        new_neuron = False
                        break
                if new_neuron == True:
                    self.w = np.vstack((self.w, [vec]))
                predicted_cluster = self.predict(np.array([vec]))
                if predicted_cluster[0] < len(del_neurons_mas):
                    del_neurons_mas[predicted_cluster[0]] = False
                self.w[predicted_cluster[0], :] += self.ls*(vec - self.w[predicted_cluster[0], :])
            for i in range(len(del_neurons_mas)):
                if(del_neurons_mas[i] == True):
                    np.delete(self.w, i, axis=0)

    def find_R(self, vec, weights):
        R = np.sqrt(np.sum(np.power(vec - weights, 2)))
        return R

    def predict(self, data):
        predicted_mas = []
        for i in range(data.shape[0]):
            boof_mas = []
            for j in range(self.w.shape[0]):
                boof_mas.append(self.find_R(data[i, :], self.w[j, :]))
            boof_mas = np.array(boof_mas)
            predicted_mas.append(np.argmin(boof_mas))
        predicted_mas = np.array(predicted_mas)
        return predicted_mas


def normalize(data):
    for j in range(data.shape[1]):
        minimum = data[:, j].min()
        maximum = data[:, j].max()
        data[:, j] = (data[:, j] - minimum)/(maximum - minimum)
    return data


data = load_breast_cancer()
X_data = data.data
y_data = data.target
X_data = normalize(X_data)

myKohonen = KohonenNetwork(number_of_clusters=0, number_of_epochs=100, learning_speed=0.01)
myKohonen.self_organization(X_data, 2.5)
y_pred = myKohonen.predict(X_data)
acc = accuracy_score(y_data, y_pred)
print(acc)
