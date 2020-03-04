import numpy as np
import math
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative(x):
    return x * (1 - x)

class Neuron:
    def __init__(self, inputs, weights, bias):
        self.delta = 0
        self.inputs = inputs
        self.weights = weights
        self.inproduct = None # Before sigmoid
        self.bias = bias
        self.new_weights = None
        self.new_bias = None

    def activation(self):
        self.inproduct = np.dot(self.inputs, self.weights) + self.bias
        return sigmoid(self.inproduct)

    def refresh_weights(self):
        self.weights = self.new_weights
        self.new_weights = None

    def refresh_bias(self):
        self.bias = self.new_bias
        self.new_bias = None

    
class NeuralNetwork:
    def __init__(self, layer_layout, learning_rate = 1):
        self.learning_rate = learning_rate
        self.input_size = 4
        self.layers = list()

        for i, neuron_count in enumerate(layer_layout):
            layer = []
            for _j in range(neuron_count):
                weight_size = self.input_size
                if i > 0:
                    weight_size = len(self.layers[i-1])
                weights = np.random.rand(1,weight_size)
                bias = np.random.random()
                neuron = Neuron(np.array([]), weights[0], bias)
                layer.append(neuron)
            self.layers.append(layer)

        print(self.layers)
    
    def train(self, values, label):
        if len(values) != self.input_size:
            raise ValueError("Input values are not the same size as NN inputs")
        if len(label) != len(self.layers[-1]):
            raise ValueError("Label values are not the same size as NN outputs")

        self.feed_forward(values)
        self.back_propagation(label)
        self.refresh_all()

    def feed_forward(self, values):
        for layer in self.layers:
            new_values = list()
            for neuron in layer:
                neuron.inputs = values
                new_values.append(neuron.activation())
            values = new_values
    
    def back_propagation(self, label):
        all_delta_weights = list()
        for i, layer in enumerate(reversed(self.layers)):
            current_weights = list()
            for j, neuron in enumerate(layer):
                if i == 0:
                    self.output_update(neuron, label[j])
                else:
                    delta_sum = 0
                    #Maybe transpose to do sum(list)
                    for neuron_weight in all_delta_weights[i-1]:
                        delta_sum += neuron_weight[j]
                    self.neuron_update(neuron, delta_sum)

                neuron_weights = list()
                for weight in neuron.weights:
                    neuron_weights.append(neuron.delta * weight)
                current_weights.append(neuron_weights)

            all_delta_weights.append(current_weights)

    def output_update(self, neuron, label):
        self.calculate_output_delta(neuron, label)
        new_weights = []
        for i, weight in enumerate(neuron.weights):
            new_weights.append(weight + (self.learning_rate * neuron.delta * neuron.inputs[i]))
        neuron.new_weights = np.array(new_weights)
        neuron.new_bias = neuron.bias + (self.learning_rate * neuron.delta)    

    def neuron_update(self, neuron, delta):
        self.calculate_neuron_delta(neuron, delta)
        new_weights = []
        for i, weight in enumerate(neuron.weights):
            new_weights.append(weight + (self.learning_rate * neuron.delta * neuron.inputs[i]))
        neuron.new_weights = np.array(new_weights)
        neuron.new_bias = neuron.bias + (self.learning_rate * neuron.delta)

    def calculate_output_delta(self, neuron, label):
        neuron.delta = (label - neuron.activation()) * derivative(sigmoid(neuron.inproduct))

    def calculate_neuron_delta(self, neuron, delta):
        if neuron.inproduct  == None:
            raise ValueError("No inporduct, something went wrong in the last steps")
        neuron.delta = (derivative(sigmoid(neuron.inproduct)) * delta)

    def refresh_all(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.refresh_bias()
                neuron.refresh_weights()
    
    def test(self, input):
        self.feed_forward(input)
        results = []
        for output_neuron in self.layers[-1]:
            results.append(output_neuron.activation())

        return results



np.random.seed(1)

converter = lambda s: [1,0,0] if s == b"Iris-setosa" else ([0,1,0] if s == b"Iris-versicolor" else [0,0,1])

#train
data_in     = np.genfromtxt('iris.data', delimiter=',', usecols=[0,1,2,3])
data_in_norm = data_in / data_in.max(axis = 0)
data_out    = np.genfromtxt('iris.data', delimiter=',', usecols=[4], converters={4: converter})
error_cutoff = 0.05

network = NeuralNetwork([2,2,3], 0.05)

for _i in range(0, 1000):
    if _i % 10 == 0:
        print(_i)
    for i, input in enumerate(data_in_norm):
        network.train(input, data_out[i])

for i, input in enumerate(data_in_norm[:2]):
    print(network.test(input))
