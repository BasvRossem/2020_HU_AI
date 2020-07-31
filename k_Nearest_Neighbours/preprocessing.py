"""Functions needed to preprocess the data."""
import numpy as np


def convert_sq_rh(value):
    """
    Convert the -1 values to a more useable 0 value.
    """
    return 0 if value == b"-1" else float(value)


def scale(x, in_min, in_max, out_min, out_max):
    """
    Scale a value to a new range of numbers.
    """
    return (x - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


def scale_data(data, new_min, new_max):
    """
    Scale all values in a dataset to a new range.
    """
    min_val = np.amin(data, 0)
    max_val = np.amax(data, 0)
    for i in range(0, len(data)):
        for j in range(1, len(data[i])):
            data[i][j] = scale(data[i][j],
                               min_val[j], max_val[j],
                               new_min, new_max)
