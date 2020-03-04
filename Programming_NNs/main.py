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
        self.is_input_layer = False
        self.is_bias = False
        self.data = 0
        self.error = 0

    # Makes a two way connection between the neurons
    def connect(self, neuron, weight):
        self.connections_forward[neuron] = weight

    def generate_output(self):
        for neuron, weight in self.connections_forward.items():
            neuron.data += self.data * weight
    
class NeuralNetwork:
    def __init__(self, learning_rate = 1):
        self.layers = list()
        
        print("Creating neural network")
        self.total_error = 0
        self.learning_rate = learning_rate

        # Create 4 input Neurons
        self.layers.append(list())
        for i in range(4):
            input_neuron = Neuron()
            input_neuron.is_input_layer = True
            self.layers[0].append(input_neuron)

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
        
        # Add bias nodes 
        for i in range(len(self.layers) - 1):
            bias = Neuron()
            bias.data = 1
            bias.is_bias = True
            for connection in self.layers[i + 1]:
                bias.connect(connection, np.random.random())
            self.layers[i].append(bias)   

        print("Creating neural network Done")      

    def train(self, training_input, training_expected, runs):
        for run in range(runs):
            self.total_error = 0
            for index in range(len(training_input)):
                # Set all data to 0
                self.reset_data() 

                # Make predictions
                self.predict(training_input[index])

                # Back propagate
                self.back_propagate(training_expected[index])

            # if run % 10 == 0:
            #     print("==========")
            #     print("Data NN:")
            #     for layer in self.layers:
            #         for neuron in layer:
            #             print(neuron.data)
            #         print("-------------")
            print("Total error:", self.total_error)

        print("Network trained")

    def predict(self, inputs):
        # Add input values to the input layer
        for neuron_index in range(len(self.layers[0]) - 1):
            self.layers[0][neuron_index].data = inputs[neuron_index]

        # Feed forward
        for i in range(len(self.layers)):
            if i != 0:
                for neuron in self.layers[i]:
                    if not neuron.is_bias:
                        neuron.data = sigmoid(neuron.data)
            for neuron in self.layers[i]:
                neuron.generate_output()
                  
    def back_propagate(self, expected):
        # Calculate the error for the neurons in the output layers
        for neuron_index in range(len(self.layers[-1])):
            neuron = self.layers[-1][neuron_index]
            neuron.error = (expected[neuron_index] - neuron.data) * derivative(neuron.data)
            self.total_error += (expected[neuron_index] - neuron.data) ** 2

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
        for layer in self.layers:
            for neuron in layer:
                for sub_neuron, weight in neuron.connections_forward.items():
                    if neuron.is_bias:
                        neuron.connections_forward[sub_neuron] = weight + (self.learning_rate * sub_neuron.error)
                    else:
                        neuron.connections_forward[sub_neuron] = weight + (self.learning_rate * sub_neuron.error * neuron.data)

    def reset_data(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.data = 0

np.random.seed(1)

converter = lambda s: [1,0,0] if s == b"Iris-setosa" else ([0,1,0] if s == b"Iris-versicolor" else [0,0,1])

#train
data_in     = np.genfromtxt('iris.data', delimiter=',', usecols=[0,1,2,3])
data_in_norm = data_in / data_in.max(axis = 0)
data_out    = np.genfromtxt('iris.data', delimiter=',', usecols=[4], converters={4: converter})
error_cutoff = 0.05

NN = NeuralNetwork(3)
NN.train(data_in_norm, data_out, 2000)

q1 = data_in_norm[0]
NN.predict(np.array(q1))

print("Expected output: ", data_out[0])
for neuron in NN.layers[-1]:
    print(neuron.error)

