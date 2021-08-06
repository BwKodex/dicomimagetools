from typing import List, Optional, Tuple, Union

from numpy import (
    abs,
    absolute,
    add,
    arctan,
    array,
    ceil,
    clip,
    divide,
    exp,
    median,
    multiply,
    ones,
    pi,
    power,
    sort,
    sqrt,
    subtract,
    sum,
    vectorize,
    zeros,
)
from scipy import linalg
from scipy.optimize import curve_fit

from .point import IntPoint, Point

LINE_COORDINATE = Union[int, float]
FIT_DATA = Union


class Line:
    def __init__(self, x0: LINE_COORDINATE, y0: LINE_COORDINATE, x1: LINE_COORDINATE, y1: LINE_COORDINATE):
        if not all([(isinstance(obj, int) or isinstance(obj, float)) for obj in [x0, y0, x1, y1]]):
            raise TypeError("All line coordinates must be either integers or floats")

        self.slope = (y1 - y0) / (x1 - x0)
        self.constant = y0 - self.slope * x0
        self.angle = arctan(self.slope) * 180 / pi


def get_shortest_line_point_dist(line: Line, point: Union[Point, IntPoint]):
    if not isinstance(line, Line):
        raise TypeError("line must be a Line object")

    if not isinstance(point, Point) and not isinstance(point, IntPoint):
        raise TypeError("point must be either a Point or IntPoint instance")

    distance = absolute(line.slope * point.x + line.constant - point.y) / sqrt(line.slope ** 2 + (-1) ** 2)
    if line.angle <= 45 or line.angle >= 135:
        if point.y < (line.slope * point.x + line.constant):
            distance = -distance
    else:
        if point.x < (point.y - line.constant) / line.slope:
            distance = -distance

    return distance


def third_deg_polynomial_fit_func(x, a, b, c, d):
    return add(multiply(a, power(x, 3)), add(multiply(b, power(x, 2)), add(multiply(c, x), d)))


def get_third_degree_polynomial_fit(x: List[Union[float, int]], y: List[Union[float, int]]):
    """Fits data to a 3rd degree polynomial on the format y = a*x^3 + b*x^2 + c*x + d. Returns y-values calculated with
    the optimized function and the z, b, c, and d parameters in a dict.

    Args:
        x: Values on the x-axis
        y: Noisy y-values to fit to the function

    Returns:
        A dictionary on with the keys a, b, c, d and y, where y is the optimized y values.

    """
    popt, pcov = curve_fit(f=third_deg_polynomial_fit_func, xdata=x, ydata=y)
    optimized_y = third_deg_polynomial_fit_func(x, *popt)

    return dict(a=popt[0], b=popt[1], c=popt[2], d=popt[3], y=optimized_y)


def _bell_kernel_function(x: Union[int, float], x0: Union[int, float], tau: Optional[float] = 0.005):
    return exp(-divide(power(subtract(x, x0), 2), multiply(2, tau)))


def _test_function(x, x1, ind):
    output = sort(abs(subtract(x, x1)))[ind]
    print(output)
    return output


def lowess(
    x: List[Union[float, int]], y: List[Union[float, int]], f: Optional[float] = 2.0 / 3.0, iter: Optional[int] = 3
):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(ceil(f * n))
    tf = vectorize
    h = [sort(abs(subtract(x, x[i])))[r] for i in range(n)]
    w = clip(abs(divide((subtract(x[:, None], x[None, :])), h)), 0.0, 1.0)
    w = power(power(subtract(1, w), 3), 3)
    yest = zeros(n)
    delta = ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = multiply(delta, w[:, i])
            b = array([sum(multiply(weights, y)), sum(multiply(weights, multiply(y, x)))])
            A = array(
                [[sum(weights), sum(multiply(weights, x))], [sum(weights * x), sum(multiply(multiply(weights, x), x))]]
            )
            beta = linalg.solve(A, b)
            yest[i] = add(beta[0], multiply(beta[1], x[i]))

        residuals = subtract(y, yest)
        s = median(abs(residuals))
        delta = clip(residuals / multiply(6.0, s), -1, 1)
        delta = power(subtract(1, power(delta, 2)), 2)

    return yest
