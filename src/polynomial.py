import math
from itertools import product

import numpy as np
import scipy.optimize as optimize


DEGREE = 2


def dfdx(a, x, y):
    res = 0.0
    for i, j in product(range(DEGREE), repeat=2):
        res += a[2 * (i * DEGREE + j)] * (2*i+1) * x**(2*i) * y**(2*j)
    return res


def dfdy(a, x, y):
    res = 0.0
    for i, j in product(range(DEGREE), range(1, DEGREE)):
        res += a[2 * (i * DEGREE + j)] * (2*j) * x**(2*i+1) * y**(2*j-1)
    return res


def dgdx(a, x, y):
    res = 0.0
    for i, j in product(range(1, DEGREE), range(DEGREE)):
        res += a[2 * (i * DEGREE + j) + 1] * (2*i) * x**(2*i-1) * y**(2*j+1)
    return res


def dgdy(a, x, y):
    res = 0.0
    for i, j in product(range(DEGREE), repeat=2):
        res += a[2 * (i * DEGREE + j) + 1] * (2*j+1) * x**(2*i) * y**(2*j)
    return res


def calc_error(a):
    N = 100
    r2 = 0.0
    w = 0.0
    for i, j in product(range(2 * N), range(N)):
        x = (i + 0.5) / (2 * N) * math.pi
        y = (j + 0.5) / (2 * N) * math.pi
        c = math.cos(y)
        w += c
        matrix = np.matrix([[dfdx(a, x, y) / c, dfdy(a, x, y)],
                            [dgdx(a, x, y) / c, dgdy(a, x, y)]])
        r2 += c * sum(np.log(np.linalg.eigvalsh(matrix.T*matrix))**2)
    return r2 / w


print(optimize.minimize(calc_error, [1, 1] + [0] * (2*DEGREE*DEGREE - 2)))
