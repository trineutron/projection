import math
from itertools import product

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


def calc_error(m):
    N = 100
    r2 = 0.0
    w = 0.0
    for i, j in product(range(2 * N), range(N)):
        x = (i + 0.5) / (2 * N) * math.pi
        y = (j + 0.5) / (2 * N) * math.pi
        cs = math.cos(y)
        w += cs
        a = dfdx(m, x, y) / cs
        b = dfdy(m, x, y)
        c = dgdx(m, x, y) / cs
        d = dgdy(m, x, y)
        tr = a*a + b*b + c*c + d*d
        det = (a*d - b*c) ** 2
        x0 = (tr + (tr*tr - 4*det)**0.5) / 2
        x1 = det / x0
        r2 += cs * (math.log(x0)**2 + math.log(x1)**2)
    return r2 / w


init = [0] * (2*DEGREE*DEGREE)
init[0] = 1
init[1] = 1
print(optimize.minimize(calc_error, init, method='Powell'))
