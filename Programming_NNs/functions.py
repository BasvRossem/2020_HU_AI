"""Functions to use in combination with the neural network."""

import numpy as np


def sigmoid(x):
    """Returns the sigmoid of the input."""
    return 1 / (1 + np.exp(-x))


def derivative(x):
    """Returns the derivative of the input."""
    return x * (1 - x)
