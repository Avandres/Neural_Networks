from random import random
import numpy as np

class RecurrentNetwork():

    def __init__(self, hidden_layer_neurons=[],
                 recurrent_layers=[],
                 learning_speed=0.001,
                 max_iter=200,
                 activ_type="ReLU"):
        self.eta = learning_speed
        self.hidden_layer_neurons = hidden_layer_neurons
        self.recurrent_layers = recurrent_layers
        self.epochs = max_iter
        self.type = activ_type

    def create_structure(self, input_layer, output_layer):
        if len(self.hidden_layer_neurons) > len(self.recurrent_layers):
            while len(self.hidden_layer_neurons) != len(self.recurrent_layers):
                self.recurrent_layers.append(0)
        elif len(self.hidden_layer_neurons) > len(self.recurrent_layers):
            while len(self.hidden_layer_neurons) != len(self.recurrent_layers):
                self.hidden_layer_neurons.append(1)
        self.neurons = []
        self.weights = []
        self.delay_neurons = []
        self.delay_weights = []
        self.gradient = []
        self.delay_gradient = []

        self.neurons.append(np.zeros(input_layer+1))
        self.neurons[-1][-1] = 1

        for l in range(len(self.hidden_layer_neurons)):
            self.weights.append(np.random.sample((self.neurons[-1].shape[0], self.hidden_layer_neurons[l])))
            self.neurons.append(np.zeros(self.hidden_layer_neurons[l] + 1))
            self.neurons[-1][-1] = 1
            self.gradient.append(np.zeros(self.hidden_layer_neurons[l]))

            if self.recurrent_layers[l] != 0:
                self.delay_neurons.append(np.zeros(self.hidden_layer_neurons[l]))
                self.delay_weights.append(np.random.sample((self.hidden_layer_neurons[l], self.hidden_layer_neurons[l])))
                self.delay_gradient.append(np.zeros(self.hidden_layer_neurons[l]))
            else:
                self.delay_neurons.append(None)
                self.delay_weights.append(None)
                self.delay_gradient.append(None)

        self.weights.append(np.random.sample((self.neurons[-1].shape[0], output_layer)))
        self.neurons.append(np.zeros(output_layer))
        self.answers = np.zeros(output_layer)
        self.gradient.append(np.zeros(output_layer))



    def fit(self, X_train, y_train):
        self.create_structure(X_train.shape[1], np.unique(y_train).shape[0])
        for e in range(self.epochs):
            for ind in range(X_train.shape[0]):
                y_code = self.target_to_code(y_train[ind], output_layer_shape=self.neurons[-1].shape[0])
                self.predict(np.array([X_train[ind]]))
                self.backpropagation_through_time(y_code)
            for l in range(len(self.hidden_layer_neurons)):
                if self.recurrent_layers[l] !=0:
                    for i in range(self.delay_neurons[l].shape[0]):
                        self.delay_neurons[l][i] = 0


    def backpropagation_through_time(self, y_code):
        for i in range(self.neurons[-1].shape[0]):
            y_error = y_code[0, i] - self.neurons[-1][i]
            self.gradient[-1][i] = self.eta*y_error*self.activation_function(self.neurons[-1][i], type=self.type, p=True)
        for l in range(len(self.neurons)-2, 0, -1):
            for i in range(self.neurons[l].shape[0]-1):
                reverse_z = self.activation_function(self.neurons[l][i], type=self.type, p=True)*self.weights[l][i, :]
                reverse_z = np.sum(np.dot(self.gradient[l], reverse_z))
                self.gradient[l-1][i] = reverse_z
        for l in range(len(self.hidden_layer_neurons)):
            if self.recurrent_layers[l] !=0:
                for i in range(self.delay_neurons[l].shape[0]):
                    reverse_z = self.activation_function(self.delay_neurons[l][i], type=self.type, p=True) * self.delay_weights[l][i, :]
                    reverse_z = np.sum(np.dot(reverse_z, self.gradient[l]))
                    self.delay_gradient[l][i] = reverse_z

        for l in range(len(self.neurons)-2):
            for i in range(self.neurons[l].shape[0]):
                for j in range(self.neurons[l+1].shape[0]-1):
                    gradient = self.neurons[l][i]*self.gradient[l][j]
                    self.weights[l][i, j] += gradient
        for i in range(self.neurons[-2].shape[0]):
            for j in range(self.neurons[-1].shape[0]):
                gradient = self.neurons[-2][i] * self.gradient[-1][j]
                self.weights[-1][i, j] += gradient
        for l in range(len(self.hidden_layer_neurons)):
            if self.recurrent_layers[l] !=0:
                for i in range(self.delay_neurons[l].shape[0]):
                    for j in range(self.neurons[l+1].shape[0] - 1):
                        gradient = self.delay_neurons[l][i]*self.gradient[l][j]
                        self.delay_weights[l][i, j] += gradient
        for l in range(len(self.hidden_layer_neurons)):
            if self.recurrent_layers[l] !=0:
                for i in range(self.delay_neurons[l].shape[0]):
                    for j in range(self.neurons[l+1].shape[0] - 1):
                        gradient = self.delay_neurons[l][i]*self.delay_gradient[l][j]
                        self.delay_weights[l][i, j] += gradient



    def target_to_code(self, target, output_layer_shape):
        code = np.zeros((1, output_layer_shape))
        code[0, target] = 1
        return code

    def activation_function(self, z, type="logistic", p=False):
        if type == "logistic":
            if not p:
                return 1/(1+np.exp(-z))
            else:
                return (1/(1+np.exp(-z)))*(1-(1/(1+np.exp(-z))))
        elif type == "ReLU":
            if not p:
                zero_mas = np.zeros(z.shape[0])
                return np.maximum(zero_mas, z)
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


    def z_input(self, x_mas, weights):
        return np.sum(np.dot(x_mas, weights))


    def predict(self, X_data):
        y_pred = []
        for data in X_data:
            self.neurons[0][:-1] = data
            if self.hidden_layer_neurons !=[]:
                for i in range(self.neurons[1].shape[0]-1):
                    self.neurons[1][i] = self.z_input(self.neurons[0], self.weights[0][:, i])
                    if self.recurrent_layers[0] != 0:
                        z_mas = self.activation_function(self.delay_neurons[0], type=self.type)
                        self.neurons[1][i] += self.z_input(z_mas, self.delay_weights[0][:, i])
            for l in range(1, len(self.hidden_layer_neurons)):
                for i in range(self.neurons[l+1].shape[0]-1):
                    z_mas = self.activation_function(self.neurons[l][:-1], type=self.type)
                    self.neurons[l+1][i] = self.z_input(z_mas, self.weights[l][:-1, i]) + \
                                           self.neurons[l][-1]*self.weights[l][-1, i]
                    if self.recurrent_layers[l] != 0:
                        z_mas = self.activation_function(self.delay_neurons[l], type=self.type)
                        self.neurons[l + 1][i] += self.z_input(z_mas, self.delay_weights[l][:, i])

            for i in range(self.neurons[-1].shape[0]):
                z_mas = self.activation_function(self.neurons[-2][:-1], type=self.type)
                self.neurons[-1][i] = self.neurons[-2][-1]*self.weights[-1][-1, i] + \
                                      self.z_input(z_mas, self.weights[-1][:-1, i])
            self.answers = self.activation_function(self.neurons[-1], type=self.type)
            y_pred.append(np.argmax(self.answers))
            for l in range(len(self.hidden_layer_neurons)):
                if self.recurrent_layers[l] != 0:
                    self.delay_neurons[l] = self.neurons[l+1][:-1]
        return np.array(y_pred)




X_train = np.array([[1], [0], [1], [1], [0], [0], [0], [0], [1], [1], [0], [1], [0], [0], [1], [0], [1], [1], [0], [1]])
y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0])

myNN = RecurrentNetwork(hidden_layer_neurons=[3],
                        recurrent_layers=[1],
                        max_iter=1000, learning_speed=0.05, activ_type="logistic")
myNN.fit(X_train, y_train)
print(myNN.predict(X_train))
print(y_train)
