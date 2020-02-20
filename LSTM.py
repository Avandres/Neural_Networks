import numpy as np
from random import random


class LSTM():

    def __init__(self,
                 hidden_layer_size,
                 max_iter=200,
                 init_speed=0.001,
                 t_steps=3):
        self.hidden_layer_size = hidden_layer_size
        self.max_iter = max_iter
        self.eta = init_speed
        self.t_steps = t_steps
        self.hidden_layers_activ_type = "tanh"
        self.logic_nn_activ_type = "logistic"


    def create_structure(self, input_layer_size, output_layer_size):
        self.neurons = []
        self.weights = []
        self.C = []
        self.h = []
        self.delay_weights = []
        self.gradient = []
        self.delay_gradient = []
        self.hidden_memory = []

        self.neurons.append(np.zeros(input_layer_size+1))
        self.neurons[-1][-1] = 1

        self.f_inputs = []
        self.f_weights = []
        self.f_z_output = []
        self.f_function_output = []

        self.i_inputs = []
        self.i_weights = []
        self.i_z_output = []
        self.i_function_output = []

        self.o_inputs = []
        self.o_weights = []
        self.o_z_output = []
        self.o_function_output = []

        for l in range(len(self.hidden_layer_size)):
            self.weights.append(np.random.sample((self.neurons[-1].shape[0], self.hidden_layer_size[l])))
            self.neurons.append(np.zeros(self.hidden_layer_size[l] + 1))
            self.neurons[-1][-1] = 1
            self.gradient.append(np.zeros(self.hidden_layer_size[l]))

            self.C.append([])
            self.h.append([])
            self.hidden_memory.append([])
            self.delay_weights.append(np.random.sample((self.hidden_layer_size[l], self.hidden_layer_size[l])))
            self.delay_gradient.append(np.zeros(self.hidden_layer_size[l]))

            self.f_inputs.append([])
            self.f_weights.append(np.random.sample(input_layer_size + self.hidden_layer_size[l]+1))
            self.f_z_output.append([])
            self.f_function_output.append([])

            self.i_inputs.append([])
            self.i_weights.append(np.random.sample(input_layer_size + self.hidden_layer_size[l]+1))
            self.i_z_output.append([])
            self.i_function_output.append([])

            self.o_inputs.append([])
            self.o_weights.append(np.random.sample(input_layer_size + self.hidden_layer_size[l]+1))
            self.o_z_output.append([])
            self.o_function_output.append([])

            for t in range(self.t_steps):
                self.C[l].append(np.zeros(self.hidden_layer_size[l]))
                self.h[l].append(np.zeros(self.hidden_layer_size[l]))
                self.hidden_memory[l].append(np.zeros(self.hidden_layer_size[l]))


                self.f_inputs[l].append(np.zeros(input_layer_size + self.hidden_layer_size[l]+1))
                self.f_inputs[l][t][-1] = 1
                self.f_z_output[l].append(0)
                self.f_function_output[l].append(0)

                self.i_inputs[l].append(np.zeros(input_layer_size + self.hidden_layer_size[l]+1))
                self.i_inputs[l][t][-1] = 1
                self.i_z_output[l].append(0)
                self.i_function_output[l].append(0)

                self.o_inputs[l].append(np.zeros(input_layer_size + self.hidden_layer_size[l]+1))
                self.o_inputs[l][t][-1] = 1
                self.o_z_output[l].append(0)
                self.o_function_output[l].append(0)

        self.weights.append(np.random.sample((self.neurons[-1].shape[0], output_layer_size)))
        self.neurons.append(np.zeros(output_layer_size))
        self.answers = np.zeros(output_layer_size)
        self.gradient.append(np.zeros(output_layer_size))


    def fit(self, X_train, y_train):
        self.create_structure(X_train.shape[1], np.unique(y_train).shape[0])
        for e in range(self.max_iter):
            for ind in range(X_train.shape[0]):
                y_code = self.target_to_code(y_train[ind], output_layer_shape=self.neurons[-1].shape[0])
                self.predict(np.array([X_train[ind]]))
                self.backpropagation_through_time(y_code)
            for l in range(len(self.hidden_layer_size)):
                for t in range(self.t_steps):
                    self.h[l][t] = np.zeros(self.hidden_layer_size[l])
                    self.C[l][t] = np.zeros(self.hidden_layer_size[l])


    def backpropagation_through_time(self, y_code):
        for i in range(self.neurons[-1].shape[0]):
            y_error = y_code[0, i] - self.neurons[-1][i]
            self.gradient[-1][i] = self.eta*y_error*self.activation_function(self.neurons[-1][i], type=self.hidden_layers_activ_type, p=True)
        for l in range(len(self.neurons)-2, 0, -1):
            for i in range(self.neurons[l].shape[0]-1):
                reverse_z = self.activation_function(self.neurons[l][i], type=self.hidden_layers_activ_type, p=True)*self.weights[l][i, :]
                reverse_z = np.sum(np.dot(self.gradient[l], reverse_z))
                self.gradient[l-1][i] = reverse_z

        for l in range(len(self.hidden_layer_size)):
            for i in range(self.hidden_layer_size[l]):
                grad_sum = 0
                for j in range(self.hidden_layer_size[l]):
                    grad_sum += self.delay_weights[l][j, i] * self.C[l][-2][j]
                self.delay_gradient[l][i] = grad_sum * self.gradient[l][i]

            f_grad = np.sum(self.delay_gradient[l])
            f_grad = f_grad * self.o_function_output[l][-2] * self.activation_function(self.f_z_output[l][-2],
                                                                                    type=self.logic_nn_activ_type, p=True)

            for i in range(self.hidden_layer_size[l]):
                grad_sum = 0
                for j in range(self.hidden_layer_size[l]):
                    grad_sum += self.delay_weights[l][j, i] * self.hidden_memory[l][-2][j]
                self.delay_gradient[l][i] = grad_sum * self.gradient[l][i]

            i_grad = np.sum(self.delay_gradient[l])
            i_grad = i_grad * self.o_function_output[l][-2] * self.activation_function(self.i_z_output[l][-2],
                                                                                    type=self.logic_nn_activ_type, p=True)
            for i in range(self.hidden_layer_size[l]):
                grad_sum = 0
                for j in range(self.hidden_layer_size[l]):
                    grad_sum += self.delay_weights[l][j, i] * self.hidden_memory[l][-2][j] * self.i_function_output[l][-2] * \
                                self.activation_function(self.o_z_output[l][-2], type=self.logic_nn_activ_type,
                                                         p=True)
                    grad_sum += self.delay_weights[l][j, i] * self.C[l][-2][j] * self.f_function_output[l][-2] * \
                                self.activation_function(self.o_z_output[l][-2], type=self.logic_nn_activ_type,
                                                         p=True)
                self.delay_gradient[l][i] = grad_sum * self.gradient[l][i]

            o_grad = np.sum(self.delay_gradient[l])

            for i in range(self.f_weights[l].shape[0]):
                self.f_weights[l][i] += f_grad * self.f_inputs[l][-2][i]
            for i in range(self.i_weights[l].shape[0]):
                self.i_weights[l][i] += i_grad * self.i_inputs[l][-2][i]
            for i in range(self.o_weights[l].shape[0]):
                self.o_weights[l][i] += o_grad * self.o_inputs[l][-2][i]

        if self.t_steps > 2:
            self.gates_optimization()

        for l in range(len(self.neurons)-2):
            for i in range(self.neurons[l].shape[0]):
                for j in range(self.neurons[l+1].shape[0]-1):
                    gradient = self.neurons[l][i]*self.gradient[l][j]
                    self.weights[l][i, j] += gradient
        for i in range(self.neurons[-2].shape[0]):
            for j in range(self.neurons[-1].shape[0]):
                gradient = self.neurons[-2][i] * self.gradient[-1][j]
                self.weights[-1][i, j] += gradient
        for l in range(len(self.hidden_layer_size)):
            for i in range(self.h[l][-2].shape[0]):
                for j in range(self.neurons[l+1].shape[0] - 1):
                    gradient = self.h[l][-2][i]*self.gradient[l][j]
                    self.delay_weights[l][i, j] += gradient

    def gates_optimization(self):
        for t in range(3, self.t_steps):
            for l in range(len(self.hidden_layer_size)):
                for i in range(self.hidden_layer_size[l]):
                    grad_sum = 0
                    for j in range(self.hidden_layer_size[l]):
                        grad_sum += self.delay_weights[l][j, i] * self.recursive_gate_optimization(3, t, l, j, "f")
                    self.delay_gradient[l][i] = grad_sum * self.gradient[l][i]

                f_grad = np.sum(self.delay_gradient[l])
                f_grad = f_grad * self.o_function_output[l][-t] * self.f_z_output[l][-t]

                for i in range(self.hidden_layer_size[l]):
                    grad_sum = 0
                    for j in range(self.hidden_layer_size[l]):
                        grad_sum += self.delay_weights[l][j, i] * self.recursive_gate_optimization(3, t, l, j, "i")
                    self.delay_gradient[l][i] = grad_sum * self.gradient[l][i]

                i_grad = np.sum(self.delay_gradient[l])
                i_grad = i_grad * self.o_function_output[l][-t] * self.i_z_output[l][-t]


                for i in range(self.f_weights[l].shape[0]):
                    self.f_weights[l][i] += f_grad * self.f_inputs[l][-t][i]
                for i in range(self.i_weights[l].shape[0]):
                    self.i_weights[l][i] += i_grad * self.i_inputs[l][-t][i]

    def recursive_gate_optimization(self, t, max_t, l, ind, gate_type):
        if t < max_t:
            C = self.recursive_gate_optimization(t+1, max_t, l, ind, gate_type)*self.f_function_output[l][-t] + \
                   self.hidden_memory[l][-t][ind]*self.i_function_output[l][-t]
        else:
            if gate_type == "f":
                C = self.C[l][-t][ind]*self.activation_function(self.f_z_output[l][-t], type=self.logic_nn_activ_type, p=True)
            elif gate_type == "i":
                C = self.hidden_memory[l][-t][ind] * self.activation_function(self.i_z_output[l][-t],
                                                                              type=self.logic_nn_activ_type,
                                                                              p=True)
        return C


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
                if isinstance(z, np.ndarray):
                    zero_mas = np.zeros(z.shape[0])
                    return np.maximum(zero_mas, z)
                else:
                    return max(0, z)
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


    def predict(self, X_data, new_predict = False):
        y_pred = []
        if new_predict:
            for l in range(len(self.hidden_layer_size)):
                for t in range(self.t_steps):
                    self.h[l][t] = np.zeros(self.hidden_layer_size[l])
                    self.C[l][t] = np.zeros(self.hidden_layer_size[l])
        for data in X_data:
            self.neurons[0][:-1] = data
            if self.hidden_layer_size != []:
                for i in range(self.neurons[1].shape[0] - 1):
                    self.neurons[1][i] = self.z_input(self.neurons[0], self.weights[0][:, i])
                    z_mas = self.activation_function(self.h[0][-1], type=self.hidden_layers_activ_type)
                    self.neurons[1][i] += self.z_input(z_mas, self.delay_weights[0][:, i])
            for l in range(1, len(self.hidden_layer_size)):
                for i in range(self.neurons[l + 1].shape[0] - 1):
                    z_mas = self.activation_function(self.neurons[l][:-1], type=self.hidden_layers_activ_type)
                    self.neurons[l + 1][i] = self.z_input(z_mas, self.weights[l][:-1, i]) + \
                                             self.neurons[l][-1] * self.weights[l][-1, i]
                    z_mas = self.activation_function(self.h[l][-1], type=self.hidden_layers_activ_type)
                    self.neurons[l + 1][i] += self.z_input(z_mas, self.delay_weights[l][:, i])
            for i in range(self.neurons[-1].shape[0]):
                z_mas = self.activation_function(self.neurons[-2][:-1], type=self.hidden_layers_activ_type)
                self.neurons[-1][i] = self.neurons[-2][-1] * self.weights[-1][-1, i] + \
                                      self.z_input(z_mas, self.weights[-1][:-1, i])
            self.answers = self.activation_function(self.neurons[-1], type=self.hidden_layers_activ_type)


            for l in range(len(self.hidden_layer_size)):
                for t in range(self.t_steps-1):
                    self.f_inputs[l][t] = self.f_inputs[l][t+1]
                    self.f_z_output[l][t] = self.f_z_output[l][t+1]
                    self.f_function_output[l][t] = self.f_function_output[l][t+1]

                    self.i_inputs[l][t] = self.i_inputs[l][t+1]
                    self.i_z_output[l][t] = self.i_z_output[l][t+1]
                    self.i_function_output[l][t] = self.i_function_output[l][t+1]

                    self.o_inputs[l][t] = self.o_inputs[l][t+1]
                    self.o_z_output[l][t] = self.o_z_output[l][t+1]
                    self.o_function_output[l][t] = self.o_function_output[l][t+1]

                    self.C[l][t] = self.C[l][t+1]
                    self.h[l][t] = self.h[l][t+1]

                    self.hidden_memory[l][t] = self.hidden_memory[l][t+1]

                self.f_inputs[l][-1][:data.shape[0]] = data
                self.f_inputs[l][-1][data.shape[0]:-1] = self.h[l][-1]
                self.f_z_output[l][-1] = self.z_input(self.f_inputs[l][-1], self.f_weights[l])
                self.f_function_output[l][-1] = self.activation_function(self.f_z_output[l][-1], type=self.logic_nn_activ_type)
                self.i_inputs[l][-1][:data.shape[0]] = data
                self.i_inputs[l][-1][data.shape[0]:-1] = self.h[l][-1]
                self.i_z_output[l][-1] = self.z_input(self.i_inputs[l][-1], self.i_weights[l])
                self.i_function_output[l][-1] = self.activation_function(self.i_z_output[l][-1], type=self.logic_nn_activ_type)
                self.o_inputs[l][-1][:data.shape[0]] = data
                self.o_inputs[l][-1][data.shape[0]:-1] = self.h[l][-1]
                self.o_z_output[l][-1] = self.z_input(self.o_inputs[l][-1], self.o_weights[l])
                self.o_function_output[l][-1] = self.activation_function(self.o_z_output[l][-1], type=self.logic_nn_activ_type)

                self.C[l][-1] = self.C[l][-1]*self.f_function_output[l][-1]
                self.C[l][-1] = self.C[l][-1]+self.hidden_memory[l][-1]*self.i_function_output[l][-1]
                self.h[l][-1] = self.C[l][-1]*self.o_function_output[l][-1]

                self.hidden_memory[l][-1] = self.neurons[l+1][:-1]

            y_pred.append(np.argmax(self.answers))
        return np.array(y_pred)

X_train = np.array([[1], [0], [1], [1], [0], [0], [0], [0], [1], [1], [0], [1], [0], [0], [1], [0], [1], [1], [0], [1]])
y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0])
'''X_train = np.array([[0, 0],
                    [1, 0],
                    [0, 0],
                    [0, 0],
                    [1, 1],
                    [0, 0],
                    [0, 0],
                    [0, 1],
                    [0, 0],
                    [0, 0]])
y_train = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0])'''

my_lstm = LSTM(hidden_layer_size=[2], max_iter=20000, init_speed=0.01, t_steps=20)
my_lstm.fit(X_train, y_train)
print(my_lstm.predict(X_train, new_predict=True))
print(y_train)

