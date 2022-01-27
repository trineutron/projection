import math
from itertools import product

import numpy as np
import scipy.optimize as optimize


DEGREE = 3


def dfdx(a, x, y):
    res = 0.0
    for i in reversed(range(DEGREE)):
        s = 0.0
        for j in reversed(range(DEGREE)):
            s *= y*y
            s += a[2 * (i * DEGREE + j)]
        res *= x*x
        res += s * (2*i+1)
    return res


def dfdy(a, x, y):
    res = 0.0
    for i in reversed(range(DEGREE)):
        s = 0.0
        for j in reversed(range(1, DEGREE)):
            s *= y*y
            s += a[2 * (i * DEGREE + j)] * (2*j)
        res *= x*x
        res += s
    return res * x * y


def dgdx(a, x, y):
    res = 0.0
    for i in reversed(range(1, DEGREE)):
        s = 0.0
        for j in reversed(range(DEGREE)):
            s *= y*y
            s += a[2 * (i * DEGREE + j) + 1]
        res *= x*x
        res += s * (2*i)
    return res * x * y


def dgdy(a, x, y):
    res = 0.0
    for i in reversed(range(DEGREE)):
        s = 0.0
        for j in reversed(range(DEGREE)):
            s *= y*y
            s += a[2 * (i * DEGREE + j) + 1] * (2*j+1)
        res *= x*x
        res += s
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


init = [7.44972894e-01,  9.56930892e-01, -7.59578774e-02,  2.25248316e-03,
        3.00215844e-02, -3.76653854e-05,  6.89920614e-03, -3.28404823e-02,
        -3.11994786e-03,  1.91051277e-02, -1.14904338e-02, -2.28201213e-03,
        6.23910277e-05,  8.95306943e-03, -1.42209568e-03, -4.73080045e-03,
        1.05498375e-03,  4.46535830e-04]
print(optimize.minimize(calc_error, init, method='Nelder-Mead'))
