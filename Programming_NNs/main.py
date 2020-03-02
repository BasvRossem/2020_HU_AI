import numpy as np

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

    def train(self, training_inputs, training_outputs, training_iterations):
        for i in range(training_iterations):
            for input_data in training_inputs:
                output_data = self.think(input_data)
                
                error = [None] * len(self.output_layer)
                for error_i in range(len(error)):
                    error[error_i] = training_outputs[i][error_i] - output_data[error_i][0]
                output_index = 0
                for neuron in self.output_layer:
                    for weight_index in range(len(neuron.weights)):
                        
                        change = error[output_index] * neuron.sigmoid_derivative(output_data[output_index])
                        neuron.weights[weight_index] += change
                    output_index += 1
            print(self.output_layer[0].weights)

    def think(self, inputs):
        # Set all input data for the input layer
        for neuron in range(len(self.input_layer)):
            self.input_layer[neuron].input = inputs[neuron]

        # Calculate the values of the output neurons
        output = np.array(np.empty([3, 1]))
        
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
training_iterations = 100 #This should eventallu go on until certain error rate


NN = NeuralNetwork()

NN.train(data_in, data_out, training_iterations)

print(NN.think(np.array([6.7,3.0,5.2,2.3])))

