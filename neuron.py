from random import uniform
import numpy as np


class Neuron:
    def __init__(self, outputs_number, index):
        print("N_id-{}|Connections-{}".format(index, outputs_number), end="   ")
        self.output_value = 0.0
        self.connections = []
        self.index = index
        self.gradient = 0
        self.momentum_rate = 0.5
        self.learning_rate = 0.05
        for output_index in range(outputs_number):
            self.connections.append({"weight": self.random_weight(), "delta": 0.0})  # ustawienie wag losowo [0-1]


    @staticmethod
    def random_weight():
        return uniform(0, 1)

    @staticmethod
    def activation(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def activation_derivative(x):
        return Neuron.activation(x) * (1 - Neuron.activation(x))

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def set_output(self, val):
        self.output_value = val

    def feed_forward(self, prev_layer):
        summed_values = 0.0
        for neuron_index in range(len(prev_layer)):
            # print("Feed layer {} to {}: from neuron {} to neuron {}".format(prev_layer_index, prev_layer_index + 1, neuron_index, self.index))
            summed_values += prev_layer[neuron_index].output_value * prev_layer[neuron_index].connections[self.index][
                "weight"]
        self.output_value = self.activation(summed_values)

    def calc_output_layer_gradient(self, target):
        self.gradient = -(target - self.output_value) * self.activation_derivative(self.output_value)
        # print("Gradient N_id-{} = {}".format(self.index, self.gradient))

    def calc_hidden_layer_gradient(self, layer_on_the_right):
        sum_dow = self.sum_differentials_of_weights(layer_on_the_right)
        self.gradient = sum_dow * self.activation_derivative(self.output_value)
        # print("Gradient N_id-{} = {}".format(self.index, self.gradient))

    def sum_differentials_of_weights(self, layer_on_the_right):
        summed_values = 0.0
        for i in range(len(layer_on_the_right) - 1):
            summed_values += layer_on_the_right[i].gradient * self.connections[i]["weight"]
        return summed_values

    def update_neuron_weights(self, layer_on_the_left):
        # alpha - momentum
        # eta - learning rate
        for i in range(len(layer_on_the_left)):
            neuron = layer_on_the_left[i]  # lecimy po wszystkich neuronach poprzedzajacej warstwy
            old_delta_weight = neuron.connections[self.index][
                "delta"]  # poprzednia zmiana wagi pomiedzy rozwazanym neuronem a i-tym neuronem z warstwy poprzedniej
            new_delta_weight = neuron.output_value * self.gradient * self.learning_rate + self.momentum_rate * old_delta_weight
            neuron.connections[self.index]["weight"] -= new_delta_weight
            neuron.connections[self.index]["delta"] = new_delta_weight
            # print("N-{} w-{} delta={}".format(neuron.index, self.index, -new_delta_weight))
