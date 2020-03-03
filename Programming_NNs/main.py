import numpy as np
import math
import random

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork():
    def __init__(self):
        self.input_layer    = [None] * 4
        self.output_layer   = [None] * 3

        for i in range(len(self.input_layer)):
            self.input_layer[i] = Neuron()
        for i in range(len(self.output_layer)):
            self.output_layer[i] = Neuron()

        for neuron in self.output_layer:
            for neu in self.input_layer:
                neuron.connect(neu, np.random.random())

    def train(self, training_inputs, training_outputs, error_cutoff):
        total_error_abs = math.inf
        while(total_error_abs > error_cutoff):                      

            # Choose random sample from the training data
            training_index = random.randint(0, len(training_inputs) - 1)
            
            input_data = training_inputs[training_index]
            expected = training_outputs[training_index]
            output_data = self.think(input_data)

            # print("Input data:", expected)
            # print("Output data:", output_data)

            # Calculate the error for each neuron
            neuron_count = 0
            for neuron in self.output_layer:
                neuron.delta = (expected[neuron_count] - output_data[neuron_count]) * sigmoid_derivative(output_data[neuron_count])
                neuron_count += 1
                #print(training_outputs[input_index][error_i], output_data[error_i][0]) 

            # Check if the model is trained using all data
            total_error_abs = 0            
            for i in range(len(training_inputs)):
                data_input = training_inputs[i]
                data_output = training_outputs[i]

                NN_output = self.think(data_input)

                for neuron_count in range(len(self.output_layer)):
                    total_error_abs += abs(data_output[neuron_count] - NN_output[neuron_count])

            total_error_abs = total_error_abs / len(training_inputs)

            print("Total error:", total_error_abs)
            # print("Output:", output_data)
            # print("Error:", *self.output_layer, sep=', ')

            # Change the weights
            output_index = 0
            for neuron in self.output_layer: 
                change = neuron.delta
                neuron.apply_weigth_change(change)
                output_index += 1

        print(self.output_layer[0].weights)

    def think(self, inputs):
        # Set all input data for the input layer
        for neuron in range(len(self.input_layer)):
            self.input_layer[neuron].input = inputs[neuron]
            self.input_layer[neuron].output = inputs[neuron]

        # Calculate the values of the output neurons
        output = [None] * len(self.output_layer)
        
        for i in range(len(self.output_layer)):
            output[i] = self.output_layer[i].get_output()
        return output

class Neuron():
    def __init__(self):
        self.connections = np.array(list())
        self.weights = np.array(list())
        self.delta = 0
        self.input = 0
        self.ouput = 0

    def __repr__(self):
        return str(self.delta)

    def connect(self, node, weight = 1):
        self.connections = np.append(self.connections, [node])
        self.weights     = np.append(self.weights,     [weight])

    def apply_weigth_change(self, change):
        for i in range(len(self.weights)):
            self.weights[i] += change

    def get_output(self):
        if len(self.connections):
            output_total = 0
            for i in range(len(self.connections)):
                product = self.connections[i].get_output() * self.weights[i]
                output_total += product
            self.output = sigmoid(output_total)
            return sigmoid(output_total)
        else:
            return self.input


np.random.seed(1)

converter = lambda s: [1,0,0] if s == b"Iris-setosa" else ([0,1,0] if s == b"Iris-versicolor" else [0,0,1])

#train
data_in     = np.genfromtxt('iris.data', delimiter=',', usecols=[0,1,2,3])
data_out    = np.genfromtxt('iris.data', delimiter=',', usecols=[4], converters={4: converter})
error_cutoff = 0.05


NN = NeuralNetwork()

NN.train(data_in, data_out, error_cutoff)

print(NN.think(np.array([5.8,4.0,1.2,0.2])))
print(NN.think(np.array([5.4,3.0,4.5,1.5])))
print(NN.think(np.array([5.6,2.8,4.9,2.0])))
