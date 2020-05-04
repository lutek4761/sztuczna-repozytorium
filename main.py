import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from neuralnet import Net


def run():
    # formatowanie danych
    df = pd.read_csv("winequality-red.csv", sep=';', decimal=",", dtype=np.float)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values
    y = OneHotEncoder().fit_transform(y).toarray().astype(float)  # rozbicie outputu na klasy (1 do n)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)  # wyciagniecie 20% danych do walidacji procesu uczenia

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)  # normalizacja inputow
    x_test = scaler.fit_transform(x_test)

    # dopasowanie topologi sieci do ilosci inptow
    input_layer_size = np.size(x_train, 1)
    output_layer_size = np.size(y_train, 1)
    hidden_layer_size = int((input_layer_size + output_layer_size) / 2)
    topology = [input_layer_size, hidden_layer_size, hidden_layer_size,  output_layer_size]
    #print("Topology: {}".format(topology))

    my_net = Net(topology)
    for i in range(20):
        print("Epoch {}".format(i))
        my_net.train(x_train, y_train)
        my_net.calc_total_error_of_one_epoch()
    plt.plot(range(len(my_net.overall_error_table)), my_net.overall_error_table)
    plt.xlabel("Epoch")
    plt.ylabel("RMS_error")
    plt.show()

    # test
    # my_net = Net([2, 2, 2])
    # for i in range(100):
    #     my_net.feed_forward([0.2, 0.1]) #<-input
    #     my_net.back_propagate([0.01, 0.99]) #<-target
    #     my_net.feed_forward([0.8, 0.7])  # <-input
    #     my_net.back_propagate([0.55, 0.12])  # <-target


run()
