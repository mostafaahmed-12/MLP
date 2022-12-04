#
import numpy as np
import pandas as pd

import preprocessing
import LLayer


class Network:
    def __init__(self, learning_rate, number_of_hiden_layers, number_of_neuron_in_each_hidden_laye: [], actv, bias):
        self.learning_rate = learning_rate
        self.layers = []
        self.number_of_hiden_layers = number_of_hiden_layers
        self.number_of_neuron_in_each_hidden_laye = number_of_neuron_in_each_hidden_laye
        self.actv = actv
        self.bias = bias

    def construct_network(self, row):
        frist_layer = LLayer.layer(len(row), self.number_of_neuron_in_each_hidden_laye[0], self.actv)
        frist_layer.forward(row, self.bias)
        self.layers.append(frist_layer)
        for i in range(1, self.number_of_hiden_layers):
            l = LLayer.layer(len(self.layers[i - 1].output), self.number_of_neuron_in_each_hidden_laye[i], self.actv)
            l.forward(self.layers[i - 1].output, self.bias)
            self.layers.append(l)
        output_layer = LLayer.layer(len(self.layers[len(self.layers) - 1].output), 3, self.actv)
        output_layer.forward(self.layers[len(self.layers) - 1].output, self.bias)
        self.layers.append(output_layer)

    def sigma_output_layer(self, target):
        self.layers[len(self.layers) - 1].sima = (target - self.layers[len(self.layers) - 1].output) * self.layers[
            len(self.layers) - 1].dertive_actvation()

    def forward(self, row):
        self.layers[0].forward(row, self.bias)
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i - 1].output, self.bias)

    def back_propgation(self, target):
        self.sigma_output_layer(target)
        for i in reversed(range(0, len(self.layers) - 1)):
            self.layers[i].sima = np.dot(self.layers[i + 1].sima, self.layers[i + 1].weights.T) * self.layers[i].dertive_actvation()

    def update_weights(self, row):

        sima = self.layers[0].sima
        input = np.array(row).reshape(1, row.shape[0])
        e = sima * input.T
        self.layers[0].weights = self.layers[0].weights + self.learning_rate * e
        for i in range(1, len(self.layers)):
            sima = self.layers[i].sima
            input = np.array(self.layers[i - 1].output).reshape(1, self.layers[i - 1].output.shape[0])
            e = sima * input.T
            self.layers[i].weights = self.layers[i].weights + self.learning_rate * e


def print_information_of_nural_network(mlu):
    for i in range(0, len(mlu.layers)):
        print("layer " + str(i + 1) + " information")
        if i == 0:
            print("inputs : " + str(preprocessing.x_traing[0]))
        else:
            print("inputs : " + str(mlu.layers[i - 1].output))
        print("weights : " + str(mlu.layers[i].weights))
        print("output : " + str(mlu.layers[i].output))
        print("delta : " + str(mlu.layers[i].sima))
        print("bias : " + str(mlu.layers[i].bias))

        print("-------------------------------------------------------------------------")


def normlization(target):
    ac = np.zeros([1, 3])
    if target == 1:
        ac[0][0] = 1
    elif target == 2:
        ac[0][1] = 1
    elif target == 3:
        ac[0][2] = 1
    return ac


def get_max_index(p):
    a = -1
    j = 0
    for i in range(0, 3):
        if p[i] >= a:
            a = p[i]
            j = i
    return j + 1


def run(number_of_hidden_layers, learing_rate, neurons, epochs, actvation_function, bias_bolean):
    mlu = Network(learing_rate, number_of_hidden_layers, neurons, actvation_function, bias_bolean)
    v = normlization(preprocessing.y_training[0])
    mlu.construct_network(preprocessing.x_traing[0])
    mlu.back_propgation(v)
    mlu.update_weights(preprocessing.x_traing[0])
    # train
    for a in range(1, epochs + 1):
        count = 0
        for b in range(len(preprocessing.x_traing)):
            mlu.forward(preprocessing.x_traing[b])
            mlu.back_propgation(normlization(preprocessing.y_training[b]))
            mlu.update_weights(preprocessing.x_traing[b])
            p = mlu.layers[-1].output
            if get_max_index(p) == preprocessing.y_training[b]:
                count += 1
        acc = (count / len(preprocessing.x_traing)) * 100.0

        print("Accuracy  after " + str(a) + " epoch " + str(round(acc, 2)) + "%")
    acount = 0
    confusion_matrix = np.zeros([3, 3])
    for c in range(len(preprocessing.x_testing)):
        mlu.forward(preprocessing.x_testing[c])
        p = mlu.layers[-1].output
        if get_max_index(p) == preprocessing.y_testing[c]:
            if get_max_index(p) == 1:
                confusion_matrix[0][0] += 1
            elif get_max_index(p) == 2:
                confusion_matrix[1][1] += 1
            elif get_max_index(p) == 3:
                confusion_matrix[2][2] += 1
            acount += 1
        elif get_max_index(p) != preprocessing.y_testing[c]:
            if get_max_index(p) == 2 and preprocessing.y_testing[c] == 1:
                confusion_matrix[0][1] += 1
            if get_max_index(p) == 3 and preprocessing.y_testing[c] == 1:
                confusion_matrix[0][2] += 1
            if get_max_index(p) == 1 and preprocessing.y_testing[c] == 2:
                confusion_matrix[1][0] += 1
            if get_max_index(p) == 3 and preprocessing.y_testing[c] == 2:
                confusion_matrix[1][2] += 1
            if get_max_index(p) == 1 and preprocessing.y_testing[c] == 3:
                confusion_matrix[2][0] += 1
            if get_max_index(p) == 2 and preprocessing.y_testing[c] == 3:
                confusion_matrix[2][1] += 1

    accur = (acount / len(preprocessing.x_testing)) * 100.0

    d = {"C1": confusion_matrix[:, 0], "C2": confusion_matrix[:, 1], "C3": confusion_matrix[:, 2]}
    print(
        "*************************************************************************************************************************************************************************")
    print("Accuracy  of testing     " + str(round(accur, 2)) + "%")
    print("confusion matrix")
    print(pd.DataFrame(d, index=["C1", "C2", "C3"]))
    print("************************Network**********************************************")
    print_information_of_nural_network(mlu)
