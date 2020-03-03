import numpy as np
import random

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
        while(True):
            error = [0] * len(self.output_layer)
                        
            runs_per_error = 10
            for i in range(runs_per_error):
                training_index = random.randint(0, len(training_inputs) - 1)
                input_data = training_inputs[training_index]
                expected = training_outputs[training_index]
                output_data = self.think(input_data)

                # print("Input data:", answer)
                # print("Output data:", output_data)
                for error_i in range(len(error)):
                    error_neuron = self.output_layer[error_i]
                    error[error_i] += (expected[error_i] - output_data[error_i]) * error_neuron.sigmoid_derivative(output_data[error_i])
                    #print(training_outputs[input_index][error_i], output_data[error_i][0]) 

            total_error_abs = abs(error[0]) + abs(error[1]) + abs(error[2])
            print("Error:", total_error_abs, error)

            for j in range(len(error)):
                error[j] = error[j] / runs_per_error          
            

            if total_error_abs < error_cutoff:
                print(self.output_layer[0].weights)
                break

            output_index = 0
            for neuron in self.output_layer:
                for weight_index in range(len(neuron.weights)):
                    neuron_error = error[output_index] 
                    change = 0.5 * neuron_error * input_data[output_index]
                    neuron.weights[weight_index] += change
                output_index += 1
            #print(self.output_layer[0].weights)

    def think(self, inputs):
        # Set all input data for the input layer
        for neuron in range(len(self.input_layer)):
            self.input_layer[neuron].input = inputs[neuron]

        # Calculate the values of the output neurons
        output = [None] * len(self.output_layer)
        
        for i in range(len(self.output_layer)):
            output[i] = self.output_layer[i].output()
        return output

class Neuron():
    def __init__(self):
        self.connections = np.array(list())
        self.weights = np.array(list())
        self.input = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def connect(self, node, weight = 1):
        self.connections = np.append(self.connections, [node])
        self.weights     = np.append(self.weights,     [weight])


    def output(self):
        if len(self.connections):
            output_total = 0
            for i in range(len(self.connections)):
                output_total += self.connections[i].output() * self.weights[i]
            return self.sigmoid(output_total)
        else:
            return self.input


np.random.seed(1)

converter = lambda s: [1,0,0] if s == b"Iris-setosa" else ([0,1,0] if s == b"Iris-versicolor" else [0,0,1])

#train
data_in     = np.genfromtxt('iris.data', delimiter=',', usecols=[0,1,2,3])
data_out    = np.genfromtxt('iris.data', delimiter=',', usecols=[4], converters={4: converter})
error_cutoff = 0.05 #This should eventallu go on until certain error rate


NN = NeuralNetwork()

NN.train(data_in, data_out, error_cutoff)

print(NN.think(np.array([5.8,4.0,1.2,0.2])))
print(NN.think(np.array([5.4,3.0,4.5,1.5])))
print(NN.think(np.array([5.6,2.8,4.9,2.0])))
