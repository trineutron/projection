"""Optimize projection method."""
import math
from itertools import product

from scipy import optimize


DEGREE = 2
N = 10


def calc_error(mat):
    """Calculate strain."""
    sum_error = 0.0
    weight = 0.0
    for idx_longitude, idx_latitude in product(range(2 * N), range(N)):
        longitude = (idx_longitude + 0.5) / (2 * N) * math.pi
        latitude = (idx_latitude + 0.5) / (2 * N) * math.pi
        cos_lat = math.cos(latitude)
        weight += cos_lat

        dfdx = 0.0
        dgdy = 0.0
        for i in reversed(range(DEGREE)):
            sum_dfdx = 0.0
            sum_dgdy = 0.0
            for j in reversed(range(DEGREE)):
                sum_dfdx *= latitude*latitude
                sum_dgdy *= latitude*latitude
                sum_dfdx += mat[2 * (i * DEGREE + j)]
                sum_dgdy += mat[2 * (i * DEGREE + j) + 1] * (2*j+1)
            dfdx *= longitude*longitude
            dfdx += sum_dfdx * (2*i+1)
            dgdy *= longitude*longitude
            dgdy += sum_dgdy
        dfdx /= cos_lat

        dfdy = 0.0
        for i in reversed(range(DEGREE)):
            sum_line = 0.0
            for j in reversed(range(1, DEGREE)):
                sum_line *= latitude*latitude
                sum_line += mat[2 * (i * DEGREE + j)] * (2*j)
            dfdy *= longitude*longitude
            dfdy += sum_line
        dfdy *= longitude * latitude

        dgdx = 0.0
        for i in reversed(range(1, DEGREE)):
            sum_line = 0.0
            for j in reversed(range(DEGREE)):
                sum_line *= latitude*latitude
                sum_line += mat[2 * (i * DEGREE + j) + 1]
            dgdx *= longitude*longitude
            dgdx += sum_line * (2*i)
        dgdx *= longitude * latitude / cos_lat

        trace = (dfdx*dfdx + dfdy*dfdy + dgdx*dgdx + dgdy*dgdy) / 2
        det = (dfdx*dgdy - dfdy*dgdx) ** 2
        rate0 = trace + math.sqrt(trace*trace - det)
        sum_error += cos_lat * (math.log(rate0)**2 + math.log(det/rate0)**2)
    return sum_error / weight


init = [0] * (2*DEGREE*DEGREE)
init[0] = 1
init[1] = 1
print(optimize.minimize(calc_error, init))
