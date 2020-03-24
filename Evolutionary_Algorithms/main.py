from functools import reduce
from operator import add
from operator import mul
from random import randint
from random import random
from random import seed

import numpy as np


def individual(length, min, max):
    """
    Creates an individual for a population
    :param length: the number of values in the list
    :param min: the minimum value in the list of values
    :param max: the maximal value in the list of values
    :return:
    """
    return [randint(min, max) for x in range(length)]


def population(count, length, min, max):
    return [individual(length, min, max) for x in range(count)]


def fitness(individual, target_sum, target_mult):
    """
    Determine the fitness of an individual. Lower is better.
    :param individual: the individual to evaluate
    :param target: the sum that we are aiming for (X)
    6.1 Genetic algorithms 85
    """
    piles = [[], []]
    for i, pile in enumerate(individual):
        if pile == 0:
            piles[0].append(i + 1)
        else:  # if genome is 0
            piles[1].append(i + 1)

    calculated_mult = abs(reduce(mul, piles[1], 1) - 360)
    calculated_sum = abs(reduce(add, piles[0], 0) - 36)
    return calculated_sum + calculated_mult


def grade(population, target_sum, target_mult):
    """
    Find average fitness for a population
    :param population: population to evaluate
    :param target: the value that we are aiming for (X)
    """
    summed = 0
    for x in population:
        summed += fitness(x, target_sum, target_mult)
    return summed / len(population)


def evolve(population, target_sum, target_mult, retain=0.3, random_select=0.05, mutate=0.04):
    graded = [(fitness(x, target_sum, target_mult), x) for x in population]
    graded = [x[1] for x in sorted(graded)]
    retain_length = int(len(graded) * retain)
    parents = graded[: retain_length]

    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)

    # crossover parents to create offspring
    desired_length = len(population) - len(parents)
    children = []
    while len(children) < desired_length:
        male = randint(0, len(parents) - 1)
        female = randint(0, len(parents) - 1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = int(len(male) / 2)
            child = male[: half] + female[half:]
            children.append(child)

    # mutate some individuals
    for individual in children:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual) - 1)
            individual[pos_to_mutate] = randint(
                min(individual), max(individual))

    parents.extend(children)
    return parents


target_sum = 36
target_mult = 360

p_count = 200  # number of individuals in population
i_length = 10  # N
i_min = 0  # value range for generating individuals
i_max = 1

p = population(p_count, i_length, i_min, i_max)


for i in range(500):  # we stop after 500 generations
    p = evolve(p, target_sum, target_mult)
    score = grade(p, target_sum, target_mult)
    print(score)
    if score == 0.0:
        print("Found after", i, "iterations")
        break

# ============================================================
# Variable explanation
# ============================================================
# p_count:
#   I have chosen to make a populatino of 200 individuals.
#   I have tried to increase the population count to 500,
#   but this yielded no better effects, besides a slower
#   calculation time.
#
# retain:
#   I have gotten quite a large population size, so keeping
#   a larger number of my best guys helps a lot. I do need
#   to keep in mind that i should not put the number too high
#   or else the mutations won't come through as expected.
#   
# random_select:
#   This just seems to work fine, I could not really measure
#   the impact of this one due to the randomness of the 
#   assignment.
#
# mutate:
#   I have chosen to increase to mutaion amount due to trying
#   to escape the local munimum.
# ============================================================
