import numpy as np
import matplotlib.pyplot as plt
import math
from random import choice

################################
# Data structure
################################
class Cluster:
    label = None
    center = None
    points = list()

    def calculate_center(self):
        if self.points:
            total_point = [0] * len(self.points[0])
            for point in self.points:
                for point_number in range(1, len(point)):
                    total_point[point_number] += point[point_number]

            for i in range(len(total_point)):
                total_point[i] /= len(self.points)

            self.center = total_point

    def __repr__(self):
        return "Cluster: " + str(self.label) + " center: " + str(self.center)


################################
# Functions
################################
def check_date(label):
    label = str(label)
    label = label[4:]
    label = float(label)
    
    if label < 301:
        return 'winter'
    elif 301 <= label < 601:
        return 'lente'
    elif 601 <= label < 901:
        return 'zomer'
    elif 901 <= label < 1201:
        return 'herfst'
    else: # from 01-12 to end of year
        return 'winter'

def scale(x, in_min, in_max, out_min, out_max):
    return (x - in_min) / (in_max - in_min) * (out_max - out_min) + out_min

def calculate_distance(a, b):
    distance =  \
        pow(a[1] - b[1], 2) + \
        pow(a[2] - b[2], 2) + \
        pow(a[3] - b[3], 2) + \
        pow(a[4] - b[4], 2) + \
        pow(a[5] - b[5], 2) + \
        pow(a[6] - b[6], 2) + \
        pow(a[7] - b[7], 2)
    return distance

def calculate_clusters(data_points, k, runs):
    # Create empty clusters
    clusters = list()

    # Add cluster to clusters and add center point randomly chosen
    for i in range(k):
        clusters.append(Cluster())
        clusters[i].center = choice(data_points)

    for run in range(runs):
        # Clear cluster points
        for cluster in clusters:
            cluster.points = list()

        # Add points to closest cluster center
        for point in data_points:
            closest_cluster_index = -1
            closest_cluster_distance = math.inf

            for i in range(len(clusters)):
                distance = calculate_distance(point, clusters[i].center)

                if distance < closest_cluster_distance:
                    closest_cluster_distance = distance
                    closest_cluster_index = i
                
            clusters[closest_cluster_index].points.append(point)

        # Calculate cluster center
        for cluster in clusters:
            cluster.calculate_center()

    return clusters

def calculate_intra_cluster_distance(cluster):
    total_distance = 0
    for point in cluster.points:
        total_distance += calculate_distance(point, cluster.center)

    return total_distance
        
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
        data_t[i][j] = scale(data_t[i][j], min_val[j], max_val[j], 0, 1)


min_val = np.amin(data_v, 0)
max_val = np.amax(data_v, 0)
for i in range(len(data_v)):  
    for j in range(1, len(data_v[i])):
        data_v[i][j] = scale(data_v[i][j], min_val[j], max_val[j], 0, 1)

min_val = np.amin(data_u, 0)
max_val = np.amax(data_u, 0)
for i in range(len(data_u)):  
    for j in range(1, len(data_u[i])):
        data_u[i][j] = scale(data_u[i][j], min_val[j], max_val[j], 0, 1)

################################
# Calculate clusters with given k
################################
runs = 50

cluster_sums = list()

for k in range(1, 20):
    clusters = calculate_clusters(data_t, k, runs)
    
    sum_cluster_distance = 0
    for cluster in clusters:
        intra_cluster_distance = calculate_intra_cluster_distance(cluster)
        sum_cluster_distance += intra_cluster_distance
    cluster_sums.append(sum_cluster_distance)
    print("Clusters sum: ", sum_cluster_distance)

################################
# Create image
################################
x_axis = list(range(1, len(cluster_sums)))    
plt.plot(cluster_sums)
plt.show()

################################
# K should be about 4 ish
################################
