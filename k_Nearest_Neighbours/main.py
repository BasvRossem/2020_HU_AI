import numpy as np
import math

################################
# Functions
################################
def check_date(label):
    if label < 20000301:
        return 'winter'
    elif 20000301 <= label < 20000601:
        return 'lente'
    elif 20000601 <= label < 20000901:
        return 'zomer'
    elif 20000901 <= label < 20001201:
        return 'herfst'
    else: # from 01-12 to end of year
        return 'winter'

def take_second(elem):
    return elem[1]

def scale(x, in_min, in_max, out_min, out_max):
    return (x - in_min) / (in_max - in_min) * (out_max - out_min) + out_min

def calculate_neighbours(data_t, unknown):
    distances = list()

    # Calculate distances to every point
    for data in data_t:
        distances.append([
            data,
                pow(unknown[1] - data[1], 2) + 
                pow(unknown[2] - data[2], 2) + 
                pow(unknown[3] - data[3], 2) + 
                pow(unknown[4] - data[4], 2) + 
                pow(unknown[5] - data[5], 2) + 
                pow(unknown[6] - data[6], 2) + 
                pow(unknown[7] - data[7], 2)
        ])

    distances.sort(key=take_second)
    return distances

def get_season(k, data_point):
    neighbours = data_point[-1][0:k]

    neighbour_seasons = {}

    # Check seasons of neighbours
    for elem in neighbours:
        season = check_date(elem[0][0])
        if season in neighbour_seasons:
            neighbour_seasons[season] += 1
        else:
            neighbour_seasons[season] = 1

    # Get the season with the highest value
    tmp = []
    highest_season_count = -1
    for key, value in neighbour_seasons.items():
        if value > highest_season_count:
            tmp = []
            tmp.append([key, value])
            highest_season_count = value
        elif value == highest_season_count:
            tmp.append([key, value])

    season = tmp[0][0]
    if len(tmp) > 1:
        season = check_date(neighbours[0][0][0])

    return season

def function(model_data, new_data_point):
    data_point = list()
    data_point = new_data_point.tolist()
    distances = calculate_neighbours(model_data, data_point)
    data_point.append(distances)
    return data_point


def find_k(k_min, k_max, step_size, data_t, data):
    data_v = [None] * data.shape[0] 

    for i in range(data.shape[0]):
        data_v[i] = function(data_t, data[i])
    
    best_k_list = []
    best_k_count = -1

    for k in range(k_min, k_max, step_size):
        correct_count = 0
        for validation in data_v:
            prediction = get_season(k, validation)

            if prediction == check_date(validation[0]):
                correct_count += 1 
        
        if correct_count > best_k_count:
            best_k_list = list()
            best_k_list.append(k)
            best_k_count = correct_count
        elif correct_count == best_k_count:
            best_k_list.append(k)

    return best_k_list
        
################################
# Load all data
################################
data_t = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0,1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
data_v = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0,1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
data_u = np.genfromtxt('days.csv', delimiter=';', usecols=[0,1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

################################
# Scale data to number between 0 and 100
################################
min_val = np.amin(data_t, 0)
max_val = np.amax(data_t, 0)
for i in range(len(data_t)):  
    for j in range(1, len(data_t[i])):
        data_t[i][j] = scale(data_t[i][j], min_val[j], max_val[j], 0, 100)


min_val = np.amin(data_v, 0)
max_val = np.amax(data_v, 0)
for i in range(len(data_v)):  
    for j in range(1, len(data_v[i])):
        data_v[i][j] = scale(data_v[i][j], min_val[j], max_val[j], 0, 100)

min_val = np.amin(data_u, 0)
max_val = np.amax(data_u, 0)
for i in range(len(data_u)):  
    for j in range(1, len(data_u[i])):
        data_u[i][j] = scale(data_u[i][j], min_val[j], max_val[j], 0, 100)

################################
# Calculate correct predictions with given k
################################
best_k_list = find_k(1, data_t.shape[0], 2, data_t, data_v)
best_k = best_k_list[0]
print("Best k values: ", best_k_list)

################################
# Calculate the unknown data points
################################

for unknown in data_u:
    #calculate neighbours
    unknown_data_point = function(data_t, unknown)

    # Get season prediction
    prediction = get_season(best_k, unknown_data_point)
    print(prediction)