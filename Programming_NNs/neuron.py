"""A neuron to use in a neural network."""

import numpy as np

from functions import sigmoid


class Neuron:
    """
    A simple neuron class.
    """

    def __init__(self, inputs, weights, bias, is_output = False):
        self.delta = 0
        self.inputs = inputs
        self.weights = weights
        self.inproduct = None  # Before sigmoid
        self.bias = bias

    def activation(self):
        """
        Returns the sigmoid of the inproduct of the weights and the inputs.
        """
        self.inproduct = np.dot(self.inputs, self.weights) + self.bias
        return sigmoid(self.inproduct)
