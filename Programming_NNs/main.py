"""A object oriented neural network implementation."""

import numpy as np

from neural_network import NeuralNetwork
from preprocessing import converter

np.random.seed(1234)


# read input data
data_in = np.genfromtxt('data/iris.data', delimiter=',', usecols=[0, 1, 2, 3])
data_in = data_in / data_in.max(axis=0)
labels = np.genfromtxt('data/iris.data', delimiter=',', usecols=[4], converters={4: converter})

# Creating the network
network = NeuralNetwork([4, 2, 3], 0.05)

# Training the network
ITERATIONS = 1234

for iteration in range(ITERATIONS):
    # Train
    for i, values in enumerate(data_in):
        network.train(values, labels[i])

    # Do some progress bar printing
    if (iteration % 3 == 0 and iteration != 0) or iteration == ITERATIONS-1:
        BAR_PARTS = 20
        CMD_BAR = "\r" + str(iteration + 1) + "/" + str(ITERATIONS) + "["
        bars = round(iteration / (round(ITERATIONS / BAR_PARTS)))
        for percentage in range(bars):
            CMD_BAR += "█"
        for space in range(BAR_PARTS - bars):
            CMD_BAR += "-"
        print(CMD_BAR + "]", end="", flush="True")
print("")


# Checking the network
wrong_list = []

OFFSET = 3
for i, values in enumerate(data_in[::OFFSET]):
    answered = [round(num) for num in network.predict(values)]
    expected = [float(num) for num in labels[i * OFFSET]]

    if answered != expected:
        print("Expected:", expected)
        print("Answered:", answered)
        wrong_list.append(values)

print(list(wrong_list))
