import math
from itertools import product

import scipy.optimize as optimize


DEGREE = 4


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
init = [9.99373704e-01,  9.99944801e-01, -6.47501806e-04, -8.27904126e-05,
        -9.12085230e-04, -7.42631141e-05, -1.47656012e-03,  3.36275408e-05,
        -2.36139195e-03, -9.88391533e-05, -2.56285619e-03, -2.64856987e-04,
        -3.63269268e-03, -3.72127788e-04, -6.88206064e-03, -5.39290499e-04,
        -3.51450339e-03,  1.38849796e-03, -1.89826478e-03,  1.24646431e-03,
        2.39429400e-03,  1.62083743e-03,  6.12881176e-04, -7.46912111e-04,
        3.31286266e-04,  6.49944342e-04, -2.68210867e-05, -8.51227659e-04,
        -1.30053056e-04,  9.38504669e-05, -2.00232358e-05,  3.77257136e-05]
print(optimize.minimize(calc_error, init, method='Powell'))
