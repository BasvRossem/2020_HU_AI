"""A object oriented neural network implementation."""

import time

import numpy as np

from neural_network import NeuralNetwork
from preprocessing import converter

np.random.seed(1)


# read input data
data_in = np.genfromtxt('data/iris.data', delimiter=',', usecols=[0, 1, 2, 3])
data_in = data_in / data_in.max(axis=0)
labels = np.genfromtxt('data/iris.data', delimiter=',', usecols=[4], converters={4: converter})

# Creating the network
network = NeuralNetwork([4, 2, 3], 0.05)

# Training the network
ITERATIONS = 225
start = time.perf_counter()

for iteration in range(ITERATIONS):
    if iteration % 10 == 0:
        print(iteration)
    for i, values in enumerate(data_in):
        network.train(values, labels[i])

stop = time.perf_counter()

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

print(wrong_list)
print("Amount of wrong:", len(wrong_list))

print("Training took:", stop - start, "seconds")
print("Average 1 iteration:", (stop - start) / ITERATIONS, "seconds")

print("================================================================")
guess = network.predict([5.1, 3.5, 1.4, 0.2])
print(guess)
print("================================================================")
