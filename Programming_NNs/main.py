import numpy as np
import math
import random
import time

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

    def activation(self):
        self.inproduct = np.dot(self.inputs, self.weights) + self.bias
        return sigmoid(self.inproduct)


    
class NeuralNetwork:
    def __init__(self, layer_layout, learning_rate = 1):
        self.learning_rate = learning_rate
        self.input_size = layer_layout[0]
        self.layers = list()

        for i, neuron_count in enumerate(layer_layout[1:]):
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
                    self.calculate_output_delta(neuron, label[j])
                else:
                    delta_sum = np.sum(np.array(all_delta_weights[i-1]), axis = 0)[j]
                    self.calculate_neuron_delta(neuron, delta_sum)
                self.calculate_weights(neuron)

                neuron_weights = list()
                for weight in neuron.weights:
                    neuron_weights.append(neuron.delta * weight)
                current_weights.append(neuron_weights)

            all_delta_weights.append(current_weights)
    
    def calculate_weights(self, neuron):
        new_weights = []
        for i, weight in enumerate(neuron.weights):
            new_weights.append(weight + (self.learning_rate * neuron.delta * neuron.inputs[i]))
        neuron.weights = np.array(new_weights)
        neuron.bias = neuron.bias + (self.learning_rate * neuron.delta)

    def calculate_output_delta(self, neuron, label):
        neuron.delta = (label - neuron.activation()) * derivative(sigmoid(neuron.inproduct))

    def calculate_neuron_delta(self, neuron, delta):
        if neuron.inproduct  == None:
            raise ValueError("No inporduct, something went wrong in the last steps")
        neuron.delta = (derivative(sigmoid(neuron.inproduct)) * delta)

    
    def predict(self, input):
        self.feed_forward(input)
        return self.get_output()
        
    def get_output(self):
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

network = NeuralNetwork([4,2,3], 0.05)


iterations = 225
start = time.perf_counter()

for _i in range(0, iterations):
    if _i % 10 == 0:
        print(_i)
    for i, input in enumerate(data_in_norm):
        network.train(input, data_out[i])

stop = time.perf_counter()

wrong_list = []

offset = 3
for i, input in enumerate(data_in_norm[::offset]):
    answered = [round(num) for num in network.predict(input)]
    expected = [float(num) for num in data_out[i * offset]]
    
    if answered != expected:
        print("Expected:", expected)
        print("Answered:", answered)
        wrong_list.append(input)

print(wrong_list)
print("Amount of wrong:", len(wrong_list))

print("Training took:", stop - start, "seconds")
print("Average 1 iteration:", (stop - start) / iterations, "seconds")