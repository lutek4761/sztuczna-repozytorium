from neuron import Neuron
import numpy as np


class Net:
    def __init__(self, topology):
        print("Creating neural networks structure: ")
        self.layers = []
        self.error_table = []  # dodawane sa bledy pojedycznych probek
        self.overall_error_table = []  # dodawane sa srednie bledy epok
        self.layers_num = len(topology)
        for layer_index in range(self.layers_num):
            print("Layer_depth {} Size:{}+1:".format(layer_index, topology[layer_index]), end=" ")
            self.layers.append([])  # dodanie nowej warstwy

            for neuron_index in range(topology[layer_index] + 1):  # +1 - dodanie neuronu z biasem
                outputs_number = 0 if layer_index == len(topology) - 1 else topology[
                                                                                layer_index + 1]  # +1 uwzglednienie biasu, ostatnia warstwa 0 polaczen
                self.layers[-1].append(Neuron(outputs_number,
                                              neuron_index))  # stworzenie neuronu, nadanie liczby polaczen i jego indeksu we wartstwie do identyfikacji
            self.layers[-1][-1].set_output(1)  # ustawienie biasu na 1
            print()

    def feed_forward(self, input_values):
        # print("\n\n-----------------FEED FORWARD--------------------------")
        if len(input_values) != len(
                self.layers[
                    0]) - 1:  # zabezpieczenie - liczba inputow musi byc taka sama jak liczba neuronow wejsciowych
            raise IndexError

        for neuron_index in range(len(input_values)):  # wpisanie inputu do warsty wejsciowej
            self.layers[0][neuron_index].set_output(input_values[neuron_index])

        # do kazdego neuronu poczawszy od pierwszej warstwy ukrytej podajemy jako parametr poprzednia warstwe
        for layer_index in range(1, len(self.layers)):
            prev_layer = self.layers[layer_index - 1]

            for neuron_index in range(len(self.layers[layer_index]) - 1):
                self.layers[layer_index][neuron_index].feed_forward(prev_layer)  # metoda feed forward neuronu sumuje wartosci z poprzedniej warstwy i poddaje je dzialaniu funkcji aktywacji

        # for i in range(len(self.layers[-1]) - 1):  # wypisujemy bez biasu w outpucie
        #     print("Output {} = {}".format(i, self.layers[-1][i].output_value))
        # print("---------------------------------------------------\n\n")


    def back_propagate(self, target):
        if Neuron.random_weight() > 0.99:
            self.show_output(target)
        # print("---------------Back propagation-----------------------")
        # Obliczanie funkcji kosztu w celu prezentacji danych
        total_err = 0.0
        output_layer = self.layers[-1]
        for i in range(len(target)):
            single_neuron_error = 0.5 * (target[i] - self.layers[-1][i].output_value) ** 2
            # print("Calculating error for output N{} = {}".format(i, single_neuron_error))
            total_err += single_neuron_error
            self.error_table.append(total_err)
        # print("Total error: {}\n".format(total_err))

        # Obliczanie gradientu warstwy outputu
        # print("Calculatin Layer_depth {} gradient: ".format(len(self.layers) - 1))
        for neuron_index in range(len(target)):
            output_layer[neuron_index].calc_output_layer_gradient(target[neuron_index])

        # Obliczanie gradientu warstw ukrytych
        for i in range(len(self.layers) - 2, 0, -1):
            # print("Calculating Layer_depth {} gradient: ".format(i))
            hidden_layer = self.layers[i]
            layer_on_the_right = self.layers[i + 1]

            for neuron_index in range(len(hidden_layer)):
                hidden_layer[neuron_index].calc_hidden_layer_gradient(layer_on_the_right)

        #  aktualizacja wag
        # print("\n\nAktualizacja: ")
        for i in range(len(self.layers) - 1, 0, -1):
            # print("Layer depth: {}".format(i-1))
            layer = self.layers[i]
            layer_on_the_left = self.layers[i - 1]

            for n in range(len(layer) - 1):
                layer[n].update_neuron_weights(layer_on_the_left)

    def train(self, input_data_frame, target_output_data_frame):  # robi 1 epoch
        for i in range(np.size(input_data_frame, 0)):
            self.feed_forward(input_data_frame[i, :])
            self.back_propagate(target_output_data_frame[i, :])

    def calc_total_error_of_one_epoch(self):
        self.overall_error_table.append(sum(self.error_table)/len(self.error_table))
        self.error_table.clear()

    def show_output(self, target):
        print("Output:          ", end=" ")
        for i in range(len((self.layers[-1])) - 1):
            print("%.4f" % self.layers[-1][i].output_value, end="  ")
        print("\nDesired outuput: ", end=" ")
        for i in range(len(target)):
            print("%.4f" % target[i], end="  ")
        print("\n")

