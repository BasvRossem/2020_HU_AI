import numpy as np

def toPlant(x):
    if x <= 1:
        return "Iris-setosa"
    elif x <= 2:
        return "Iris-versicolor"
    return "Iris-virginica"



class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((4,1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for i in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

converter = lambda s: 0 if s == b"Iris-setosa" else (1 if s == b"Iris-versicolor" else 2)

data_in     = np.genfromtxt('iris.data', delimiter=',', usecols=[0,1,2,3])
data_out    = np.genfromtxt('iris.data', delimiter=',', usecols=[4], converters={4: converter})
data_out    = np.array([data_out]).T 
training_iterations = 200

for i in range(1):
    NN = NeuralNetwork()
    NN.train(data_in, data_out, training_iterations)

    correct = 0
    for i in range(len(data_in)):
        answer = toPlant(data_out[i])
        print(data_in[i])
        prediction = toPlant(NN.think(np.array([data_in[i][0],data_in[i][1],data_in[i][2],data_in[i][3]])))
        if prediction == correct:
            correct += 1

    correct = correct / len(data_in)    
    print(NN.synaptic_weights.T, correct)

NN = NeuralNetwork()
NN.train(data_in, data_out, training_iterations)

A = str(input("Input 1: "))
B = str(input("Input 2: "))
C = str(input("Input 3: "))
D = str(input("Input 4: "))

print("New input data: ", A, B, C, D)
print("NN output: ", toPlant(NN.think(np.array([A,B,C,D]))))