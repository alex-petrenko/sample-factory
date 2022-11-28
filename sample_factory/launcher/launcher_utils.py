import random
from typing import List


def seeds(num_seeds: int) -> List[int]:
    return [random.randrange(1000000, 9999999) for _ in range(num_seeds)]
