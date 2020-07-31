"""Functions used to run the K Nearerst Neighbours algorithm"""


def remove_year_from_date(date):
    """
    Removes the year from the date.
    For example, yyyymmdd becomes mmdd
    """
    return float(str(date)[4:])


def get_season_from_date(label):
    """
    Get the season winter, spring, summer, or fall from a mmdd date format.
    """
    if 301 <= label < 601:
        return 'spring'
    elif 601 <= label < 901:
        return 'summer'
    elif 901 <= label < 1201:
        return 'fall'
    else:  # from 12/01 to end of year
        return 'winter'


def calculate_neighbours(data, unknown):
    """
    Calculate the distance to all neighbours
    and return a list of these distances.
    """
    distances = [None] * len(data)

    # Calculate distances to every point
    for index, data_point in enumerate(data):
        distance = 0
        for value_index in range(1, len(unknown)):
            distance += (unknown[value_index] - data_point[value_index]) ** 2

        distances[index] = (data_point, distance)

    # Sort by the second element in the list
    distances.sort(key=lambda element: element[1])
    return distances


def combine_data_neighbours(model_data, old_data_point):
    """
    Calculates the distance to all neighbours in the
    model and combines the point and the distances.
    Returns the combines data point.
    """
    data_point = old_data_point.tolist()
    distances = calculate_neighbours(model_data, data_point)
    data_point.append(distances)
    return data_point


def predict_season(k, data_point):
    """
    Predict thhe season using the distances to the other neighbours.

    It checks what the seasons are of the closest k neigbours
    and returnsd the season which most of the neighbours have.
    """
    neighbours = data_point[-1][0:k]

    neighbour_seasons = {}

    # Check seasons of neighbours
    for elem in neighbours:
        date = remove_year_from_date(elem[0][0])
        season = get_season_from_date(date)
        if season in neighbour_seasons:
            neighbour_seasons[season] += 1
        else:
            neighbour_seasons[season] = 1

    neighbour_seasons = list(neighbour_seasons.items())
    neighbour_seasons.sort(reverse=True, key=lambda element: element[1])

    return neighbour_seasons[0][0]


def find_k(k_min, k_max, step_size, data_all, data_validation):
    """
    Returns a list of the best k values.
    """
    points = [None] * data_validation.shape[0]

    # Calculate all distances to other points
    for i in range(data_validation.shape[0]):
        points[i] = combine_data_neighbours(data_all, data_validation[i])

    # Find the best k value in the predefined range
    k_correctness = {}

    for k in range(k_min, k_max, step_size):
        correct_count = 0
        for point in points:
            prediction = predict_season(k, point)
            date = remove_year_from_date(point[0])
            if prediction == get_season_from_date(date):
                correct_count += 1

        if correct_count in k_correctness:
            k_correctness[correct_count].append(k)
        else:
            k_correctness[correct_count] = [k]

        print("K: ", k, " correct: ", correct_count)

    best_k_list = list(k_correctness.items())
    best_k_list.sort(reverse=True, key=lambda element: element[0])

    return best_k_list[0][1]
