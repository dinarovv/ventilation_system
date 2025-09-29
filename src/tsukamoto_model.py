import numpy as np

def trapmf(x, a, b, c, d):
    return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a + 1e-6), 1), (d - x) / (d - c + 1e-6)))


def trimf(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-6), (c - x) / (c - b + 1e-6)))