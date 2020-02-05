import numpy as np
import math

def checkdate(label):
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
        
################################
# Load all data
################################
data_t = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0,1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
data_v = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0,1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

################################
# Create labels
################################
for data in data_t:
    new_data = np.append(data, checkdate(data[0]))
    data = new_data


for validation in data_v:
    k = 3
    unknown_label = "UNKNOWN"
    distances = {}

    # Calculate distances to every point
    for i in range(data_t.size):
        data = data_t[i]
        print("===============")
        print(data)
        print("===============")
        distances[data_t] = (
            math.sqrt(
                pow(validation[1] - data[1], 2) + 
                pow(validation[2] - data[2], 2) + 
                pow(validation[3] - data[3], 2) + 
                pow(validation[4] - data[4], 2) + 
                pow(validation[5] - data[5], 2) + 
                pow(validation[6] - data[6], 2) + 
                pow(validation[7] - data[7], 2)
            )
        )
       
    
        
    print(checkdate(unknown_label))