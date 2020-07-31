"""An implementation of k nearest neighbours"""
import numpy as np

from algorithm import combine_data_neighbours, find_k, predict_season
from preprocessing import convert_sq_rh, scale_data

# Load all data
CONVERTERS = {5: convert_sq_rh, 7: convert_sq_rh}
DATASETS = {
    "t": "data/dataset1.csv",
    "v": "data/validation1.csv",
    "u": "data/days.csv"
}

data_t = np.genfromtxt(DATASETS["t"], delimiter=';', usecols=list(range(8)), converters=CONVERTERS)
data_v = np.genfromtxt(DATASETS["v"], delimiter=';', usecols=list(range(8)), converters=CONVERTERS)
data_u = np.genfromtxt(DATASETS["u"], delimiter=';', usecols=list(range(8)), converters=CONVERTERS)


# Scale data to number between 0 and 5
scale_data(data_t, 0, 5)
scale_data(data_v, 0, 5)
scale_data(data_u, 0, 5)

# Calculate what k has the most correct values
best_k_list = find_k(1, data_t.shape[0], 2, data_t, data_v)
best_k = best_k_list[0]
print("Best k values: ", best_k_list)

# Calculate the unknown data points
for unknown in data_u:
    # Calculate neighbours
    unknown_data_point = combine_data_neighbours(data_t, unknown)

    # Get season prediction
    prediction = predict_season(best_k, unknown_data_point)
    print(prediction)
