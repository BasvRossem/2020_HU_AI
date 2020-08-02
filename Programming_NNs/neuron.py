"""A neuron to use in a neural network."""

import numpy as np

from functions import sigmoid, derivative


class Neuron:
    """
    A simple neuron class.
    """

    def __init__(self, inputs, weights, bias, is_output=False):
        self.delta = 0
        self.inputs = inputs
        self.weights = weights
        self.inproduct = None  # Before sigmoid
        self.bias = bias
        self.is_output = is_output

    def activation(self):
        """
        Returns the sigmoid of the inproduct of the weights and the inputs.
        """
        self.inproduct = np.dot(self.inputs, self.weights) + self.bias
        return sigmoid(self.inproduct)

    def calculate_delta(self, change):
        """
        Calculate and set the new neuron delta.
        """
        self.delta = change * derivative(self.activation())
