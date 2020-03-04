import numpy as np
import math
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative(x):
    return x * (1 - x)

class Neuron:
    def __init__(self):
        self.connections_forward = dict()
        self.connections_backward = dict()
        self.data = 0
        self.error = 0

    # Makes a two way connection between the neurons
    def connect(self, neuron, weight):
        self.connections_forward[neuron] = weight
        neuron.connections_backward[self] = weight

    def generate_output(self):
        for neuron, weight in self.connections_forward.items():
            neuron.data += self.data * weight
    
class NeuralNetwork:
    def __init__(self):
        self.layers = list()
        
        # Create 4 input Neurons
        self.layers.append(list())
        for i in range(4):
            self.layers[0].append(Neuron())

        # Create two hidden layers with each two neurons
        for i in range(2):
            self.layers.append(list())
            for j in range(2):
                self.layers[-1].append(Neuron())

        # Create an output layer with 3 neurons
        self.layers.append(list())
        for i in range(3):
            self.layers[-1].append(Neuron())

        # Connect all neurons with the neurons in the next layer
        for i in range(len(self.layers) - 1):
            for neuron in self.layers[i]:
                for connection in self.layers[i + 1]:
                    neuron.connect(connection, np.random.random())

    def train(self, training_input, training_expected, runs = 1000):
        for run in range(runs):
            print(run)
            for run_input in training_input:
                # Make predictions
                self.predict(run_input)

                # Back propagate
                self.back_propagate(run_input)

    def predict(self, inputs):
        # Add input values to the input layer
        for neuron_index in range(len(self.layers[0])):
            self.layers[0][neuron_index].data = inputs[neuron_index]

        # Feed forward
        for i in range(len(self.layers)):
            for neuron in self.layers[i]:
                # For every neuron not in the first layer, first alpply the simoid function before feeding you input forward
                if(i != 0):
                    neuron.data = sigmoid(neuron.data)
                neuron.generate_output()
        
    def back_propagate(self, expected):
        # Calculate the error for the neurons in the output layers
        for neuron_index in range(len(self.layers[-1])):
            neuron = self.layers[-1][neuron_index]
            neuron.error = (expected[neuron_index] - neuron.data) * derivative(neuron.data)

        # Calculate the error for the neurons in the hidden layers
        sub_layers = self.layers[1:-1]
        sub_layers.reverse()
        for layer in sub_layers:
            for neuron in layer:
                error = 0
                for sub_neuron, weight in neuron.connections_forward.items():
                    error += sub_neuron.error * weight
                neuron.error = error * derivative(neuron.data)

        # Apply new weights to the connections
        for layer in self.layers[:-1]:
            for neuron in layer:
                for sub_neuron, weight in neuron.connections_forward.items():
                    before = neuron.connections_forward[sub_neuron] 
                    neuron.connections_forward[sub_neuron] = neuron.connections_forward[sub_neuron] + (neuron.error * neuron.data)
        for neuron in self.layers[-1]:
            print(neuron.data)
        #print(before, neuron.connections_forward[sub_neuron], neuron.error , neuron.data)

        


        


np.random.seed(5)

converter = lambda s: [1,0,0] if s == b"Iris-setosa" else ([0,1,0] if s == b"Iris-versicolor" else [0,0,1])

#train
data_in     = np.genfromtxt('iris.data', delimiter=',', usecols=[0,1,2,3])
data_out    = np.genfromtxt('iris.data', delimiter=',', usecols=[4], converters={4: converter})
error_cutoff = 0.05


NN = NeuralNetwork()
NN.train(data_in, data_out, 1)

NN.predict(np.array([5.8,4.0,1.2,0.2]))

for neuron in NN.layers[-1]:
    print(neuron.data)

