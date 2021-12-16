import random


def seeds(num_seeds):
    return [random.randrange(1000000, 9999999) for _ in range(num_seeds)]
