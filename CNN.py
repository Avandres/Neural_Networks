import numpy as np
from random import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


'''
hidden_structure - список-структура, отражающий строение нейросети (только скрытых слоёв). Всего в нём столько элементов,
сколько в нейросети слоёв. В каждом элементе содержатся:
   1) число паттернов
   2) число нейронов в паттернах (без биаса)
   3) окно (window)
   4) шаг (step) 
input_structure - список-структура, отражающий строение входного слоя. В него входят:
   1) число паттернов
   2) число нейронов в паттернах (без биаса)
   3) окно (window)
   4) шаг (step) 
number_of_outputs - количество классов. 
'''

class CNN_network():

    def __init__(self, input_structure, hidden_structure, number_of_outputs,
                 activation_type="ReLU", learning_speed=0.001, max_iter=200):
        self.layers = []
        if len(hidden_structure) > 0:
            self.layers.append(layer(input_structure[0], input_structure[1],
                                     input_structure[2], input_structure[3], hidden_structure[0][0]))
        else:
            self.layers.append(layer(input_structure[0], input_structure[1],
                                     input_structure[2], input_structure[3], 1, number_of_outputs))
        for l in range(len(hidden_structure)):
            if l == len(hidden_structure)-1:
                self.layers.append(layer(hidden_structure[l][0], hidden_structure[l][1],
                                         hidden_structure[l][2], hidden_structure[l][3], 1, number_of_outputs))
            else:
                self.layers.append(layer(hidden_structure[l][0], hidden_structure[l][1],
                                         hidden_structure[l][2], hidden_structure[l][3], hidden_structure[l+1][0]))
        self.layers.append(layer(1, number_of_outputs))
        self.activ_type = activation_type
        self.eta = learning_speed
        self.max_iter = max_iter


    def predict_sample(self, X_train_sample):
        for p1 in range(X_train_sample.shape[0]):
            X_pattern = X_train_sample[p1, :]
            self.layers[0].patterns[p1].output_neurons[:] = X_pattern
        for l in range(1, len(self.layers)):
            for p in range(len(self.layers[l].patterns)):
                for n in range(self.layers[l].patterns[p].input_neurons.shape[0]):
                    is_pre_last = self.layers[l-1].patterns[0].pre_last_layer
                    self.layers[l].patterns[p].input_neurons[n] = self.z_input(self.layers[l-1], n, p, is_pre_last)
                    self.layers[l].patterns[p].output_neurons[n] = self.activation_function(self.layers[l].patterns[p].input_neurons[n],
                                                                                            type=self.activ_type)
        answer = np.argmax(self.layers[-1].patterns[0].output_neurons)
        return answer


    def backprop(self, X_train_sample, y_sample):
        self.predict_sample(X_train_sample)
        for n in range(self.layers[-1].patterns[0].output_neurons.shape[0]):
            f_z = self.layers[-1].patterns[0].output_neurons[n]
            if n == y_sample:
                self.layers[-1].patterns[0].gradient_cells[n] = 1 - f_z
            else:
                self.layers[-1].patterns[0].gradient_cells[n] = 0 - f_z
            self.layers[-1].patterns[0].gradient_cells[n] *= self.eta * self.activation_function(
                self.layers[-1].patterns[0].input_neurons[n],
                self.activ_type, p=True)

        for l in range(len(self.layers)-2, 0, -1):
            for p1 in range(len(self.layers[l+1].patterns)):
                for n in range(self.layers[l+1].patterns[p1].gradient_cells.shape[0]):
                    for p2 in range(len(self.layers[l].patterns)):
                        start_ind = n * self.layers[l].patterns[p2].step
                        grad = self.layers[l+1].patterns[p1].gradient_cells[n]
                        for ind in range(start_ind, start_ind+self.layers[l].patterns[p2].window):
                            if self.layers[l].patterns[0].pre_last_layer:
                                weight = self.layers[l].patterns[p2].weights[p1][ind-start_ind, n]
                            else:
                                weight = self.layers[l].patterns[p2].weights[p1][ind - start_ind]
                            value = self.activation_function(self.layers[l].patterns[p2].input_neurons[ind],
                                                             self.activ_type, p=True)
                            self.layers[l].patterns[p2].gradient_cells[ind] += weight*value*grad


    def weights_optimize_step(self):
        for l in range(len(self.layers)-2, -1, -1):
            for p1 in range(len(self.layers[l + 1].patterns)):
                for n in range(self.layers[l + 1].patterns[p1].gradient_cells.shape[0]):
                    for p2 in range(len(self.layers[l].patterns)):
                        if self.layers[l].patterns[0].pre_last_layer == False:
                            start_ind = n * self.layers[l].patterns[p2].step
                            grad = self.layers[l + 1].patterns[p1].gradient_cells[n]
                            for ind in range(start_ind, start_ind + self.layers[l].patterns[p2].window):
                                delta_weight = self.layers[l].patterns[p2].output_neurons[ind]*grad
                                self.layers[l].patterns[p2].weights[p1][ind-start_ind] += delta_weight
                            delta_weight = grad
                            self.layers[l].patterns[p2].bias_weights[p1] += delta_weight
                        else:
                            grad = self.layers[l + 1].patterns[p1].gradient_cells[n]
                            for i in range(self.layers[l].patterns[p2].weights[p1].shape[0]):
                                delta_weight = self.layers[l].patterns[p2].output_neurons[i] * grad
                                self.layers[l].patterns[p2].weights[p1][i, n] += delta_weight
                            delta_weight = grad
                            self.layers[l].patterns[p2].bias_weights[p1] += delta_weight


    def predict(self, X_train):
        y_pred = []
        for X_sample in X_train:
            y_pred.append(self.predict_sample(np.array([X_sample])))
        return y_pred


    def gradient_reset(self):
        for l in range(1, len(self.layers)):
            for p in range(len(self.layers[l].patterns)):
                for g in range(self.layers[l].patterns[p].gradient_cells.shape[0]):
                    self.layers[l].patterns[p].gradient_cells[g] = 0


    def fit(self, X_train, y_train):
        for e in range(self.max_iter):
            for X_sample, y_sample in zip(X_train, y_train):
                self.backprop(np.array([X_sample]), y_sample)
                self.weights_optimize_step()
                self.gradient_reset()



    def activation_function(self, z, type="logistic", p=False):
        if type == "logistic":
            if not p:
                return 1/(1+np.exp(-z))
            else:
                return (1/(1+np.exp(-z)))*(1-(1/(1+np.exp(-z))))
        elif type == "ReLU":
            if not p:
                return np.maximum(0, z)
            else:
                if z>0:
                    return 1
                else:
                    return random()/20
        elif type == "tanh":
            if not p:
                return np.tanh(z)
            else:
                return 1 - np.power(np.tanh(z), 2)
        else:
            print("There are no function type, called '", type, "'")


    def z_input(self, layer, n, num_of_pattern, pre_last=False):
        z_sum = 0
        if pre_last == False:
            for p in layer.patterns:
                ind = n*p.step
                z_sum += np.sum(p.output_neurons[ind:ind+p.window]*p.weights[num_of_pattern]) + 1*p.bias_weights[num_of_pattern]
        else:
            for p in layer.patterns:
                z_sum += np.sum(p.output_neurons*p.weights[num_of_pattern][:, n]) + 1*p.bias_weights[num_of_pattern]
        return z_sum




class layer():

    def __init__(self, number_of_patterns, number_of_neurons, window=None,
                 step=None, number_of_next_layer_patterns=None, num_of_next_layer_neurons=None):
        self.patterns = []
        for i in range(number_of_patterns):
            self.patterns.append(pattern(number_of_neurons, window, step, number_of_next_layer_patterns, num_of_next_layer_neurons))


class pattern():

    def __init__(self, pattern_shape, window=None, step=None, number_of_next_layer_patterns=None, number_of_next_layer_neurons=None):
        self.input_neurons = np.zeros(pattern_shape)
        self.gradient_cells = np.zeros(pattern_shape)
        self.output_neurons = np.zeros(pattern_shape)
        if (window != None) and (step != None) and (number_of_next_layer_patterns != None):
            self.window = window
            self.step = step
            self.bias_weights = []
            self.weights = []
            self.pre_last_layer = False
            for i in range(number_of_next_layer_patterns):
                if number_of_next_layer_neurons != None:
                    self.weights.append((np.random.sample((pattern_shape, number_of_next_layer_neurons))+1)*0.001)
                    self.pre_last_layer = True
                else:
                    self.weights.append((np.random.sample(window)+1)*0.001)
                self.bias_weights.append((np.random.sample(1)+1)*0.001)


def load_mnist(filename):
    data = pd.read_csv(filename)
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values.astype(float)

    buf_y = []
    buf_X = []
    for i in range(3):
        counter = 0
        for j in range(y.shape[0]):
            if counter == 200:
                break
            if y[j] == i:
                buf_y.append(y[j])
                buf_X.append(X[j])
                counter += 1
    X = np.array(buf_X)
    y = np.array(buf_y)
    new_X = []
    for i in range(X.shape[0]):
        X_sample = np.reshape(X[i, :], (28, 28))
        new_X_sample = []
        for j in range(0, X_sample.shape[0]-2, 2):
            for k in range(0, X_sample.shape[1]-2, 2):
                new_X_sample.append(X_sample[j, k])
                new_X_sample.append(X_sample[j, k+1])
                new_X_sample.append(X_sample[j, k+2])
                new_X_sample.append(X_sample[j+1, k])
                new_X_sample.append(X_sample[j+1, k+1])
                new_X_sample.append(X_sample[j+1, k+2])
                new_X_sample.append(X_sample[j+2, k])
                new_X_sample.append(X_sample[j+2, k+1])
                new_X_sample.append(X_sample[j+2, k+2])
        new_X.append(new_X_sample)
    return np.array(new_X), y

#X, y = load_mnist("mnist_train.csv")
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

def load_data(X):
    new_X = []
    for i in range(X.shape[0]):
        sample = []
        X_sample = X[i]
        for j in range(X_sample.shape[0]-1):
            for k in range(X_sample.shape[1]-1):
                sample.append(X_sample[j, k])
                sample.append(X_sample[j, k+1])
                sample.append(X_sample[j+1, k])
                sample.append(X_sample[j+1, k+1])
        new_X.append(sample)
    return np.array(new_X)

y = np.array([0, 1, 1, 0, 2, 2])
X = []
number = [[0, 1, 1, 0],
          [1, 0, 0, 1],
          [1, 0, 0, 1],
          [0, 1, 1, 0]]
X.append(number)
number = [[0, 0, 1, 0],
          [0, 0, 1, 0],
          [0, 0, 1, 0],
          [0, 0, 1, 0]]

X.append(number)
number = [[0, 1, 0, 0],
          [0, 1, 0, 0],
          [0, 1, 0, 0],
          [0, 1, 0, 0]]

X.append(number)
number = [[0, 0, 1, 0],
          [0, 1, 0, 1],
          [0, 1, 0, 1],
          [0, 0, 1, 0]]

X.append(number)
number = [[0, 0, 1, 0],
          [0, 0, 0, 1],
          [0, 0, 1, 0],
          [0, 1, 1, 1]]

X.append(number)
number = [[0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 1, 0, 0],
          [1, 1, 1, 0]]
X.append(number)
X = np.array(X)

new_X = load_data(X)

CNN = CNN_network([1, 36, 4, 4], [[5, 9, 9, 0]], 3, learning_speed=0.1, max_iter=100)
CNN.fit(new_X, y)
y_pred = CNN.predict(new_X)
print(y_pred)
print(accuracy_score(y, y_pred))
