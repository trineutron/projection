"""Optimize projection method."""
import math

from numba import jit
from scipy import optimize

DEGREE = 5
N = 20
WEIGHT_S = 0.5


@jit
def calc_error_point(mat, longitude, latitude):
    """Caltulate strain."""
    latitude /= 2
    cos_lat = math.cos(latitude * math.pi)

    dfdx = 0.0
    dgdy = 0.0
    for i in range(DEGREE-1, -1, -1):
        sum_dfdx = 0.0
        sum_dgdy = 0.0
        for j in range(DEGREE-1, -1, -1):
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
    for i in range(DEGREE-1, -1, -1):
        sum_line = 0.0
        for j in range(DEGREE-1, 0, -1):
            sum_line *= latitude*latitude
            sum_line += mat[2 * (i * DEGREE + j)] * (2*j)
        dfdy *= longitude*longitude
        dfdy += sum_line
    dfdy *= longitude * latitude

    dgdx = 0.0
    for i in range(DEGREE-1, 0, -1):
        sum_line = 0.0
        for j in range(DEGREE-1, -1, -1):
            sum_line *= latitude*latitude
            sum_line += mat[2 * (i * DEGREE + j) + 1]
        dgdx *= longitude*longitude
        dgdx += sum_line * (2*i)
    dgdx *= longitude * latitude / cos_lat

    trace = (dfdx*dfdx + dfdy*dfdy + dgdx*dgdx + dgdy*dgdy) / 2
    det = max(dfdx*dgdy - dfdy*dgdx, 1e-100)
    ratio = (trace + math.sqrt(trace*trace - det*det)) / det
    return cos_lat * (WEIGHT_S * math.log(det)**2
                      + (1.0 - WEIGHT_S) * math.log(ratio)**2)


@jit
def calc_error(mat):
    """Calculate sum of strain."""
    sum_error = 0.0
    for i in range(N):
        x_i = 3 * (i + 0.5) / N
        longitude = math.tanh(math.pi / 2 * math.sinh(x_i))
        weight_x = math.cosh(x_i) / math.cosh(math.pi/2 * math.sinh(x_i))**2
        for j in range(N):
            y_i = 3 * (j + 0.5) / N
            latitude = math.tanh(math.pi / 2 * math.sinh(y_i))
            weight_y = (math.cosh(y_i)
                        / math.cosh(math.pi/2 * math.sinh(y_i))**2)
            sum_error += (weight_x * weight_y
                          * calc_error_point(mat, longitude, latitude))
    return sum_error * (3 / N)**2 * (math.pi / 2)**3


init = [0] * (2*DEGREE*DEGREE)
init[0] = 1
init[1] = 1
print(optimize.minimize(calc_error, init))
