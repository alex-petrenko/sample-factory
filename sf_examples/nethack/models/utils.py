def interleave(*args):
    return [val for pair in zip(*args) for val in pair]
