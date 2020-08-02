"""Functions used in preprocessing the iris data."""


def converter(string_label):
    """
    Convert the string label data to a list label to be
    used in the neural network.
    """
    labels = {
        b"Iris-setosa": [1, 0, 0],
        b"Iris-versicolor": [0, 1, 0],
        b"Iris-virginica": [0, 0, 1],
    }
    return labels[string_label]
