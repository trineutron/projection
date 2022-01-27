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
init = [7.07833564e-01,  9.70016914e-01, -1.64054434e-02, -3.73520474e-03,
        -1.26486021e-02, -5.57350742e-03,  2.38844435e-02,  4.09087607e-03,
        2.68312810e-02, -2.50420707e-02, -1.66424276e-02,  6.89368314e-03,
        -7.84785498e-03, -6.61244226e-04, -7.19232428e-03, -9.01966287e-04,
        -2.82692592e-03,  2.32255502e-03, -1.58919550e-03,  1.35190420e-03,
        2.44049069e-03,  1.59279511e-03,  6.13985577e-04, -7.66260275e-04,
        1.27583430e-04,  7.44194824e-04,  8.31847479e-05, -8.24847903e-04,
        -1.21695078e-04,  8.79746132e-05, -1.71470982e-05,  3.89901589e-05]
print(optimize.minimize(calc_error, init))
