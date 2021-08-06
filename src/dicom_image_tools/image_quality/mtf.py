import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import norm
from skimage.exposure import rescale_intensity
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.transform import probabilistic_hough_line

from ..helpers.geometry import Line, get_shortest_line_point_dist
from ..helpers.point import IntPoint
from ..roi.square_roi import SquareRoi

logger = logging.getLogger(__name__)


@dataclass
class MtfCurve:
    """Class for holding an MTF-curve"""

    frequency: List[int]
    mtf: np.ndarray


def mtf_bar_pattern(max_hu: float, bg: float, sd: List[float], sd_bg: float) -> List[float]:
    """Implement the method by Droege and Morin (1982) Medical Physics. Determine MTF from bar pattern data

    See wiki for a detailed explanation of the method.

    Args:
        max_hu: The maximum CT number in the bar pattern ROI
        bg: The background CT number
        sd: A list of standard deviations over the line pair patterns for the different frequencies
        sd_bg: The standard deviation of a homogeneous background ROI

    Returns:
        A list of the MTF value in the same order as the supplied standard deviations

    """
    if not isinstance(max_hu, float):
        raise TypeError("max_hu must be a float")

    if not isinstance(bg, float):
        raise TypeError("bg must be a float")

    if not isinstance(sd, List) or not any([isinstance(el, float) for el in sd]):
        raise TypeError("sd must be a list of floats")

    if not isinstance(sd_bg, float):
        raise TypeError("sd_bg must be a float")

    m0 = np.divide(np.abs(np.subtract(max_hu, bg)), 2)
    n = sd_bg
    m = [np.sqrt(i ** 2 - n ** 2) if i > n else 0 for i in sd]

    mtf = [np.multiply(np.divide(np.multiply(np.pi, np.sqrt(2.0)), 4.0), np.divide(msa, m0)) for msa in m]

    return mtf


def mtf_edge_phantom(image: np.ndarray, roi: SquareRoi, acceptable_line_gap: int = 20) -> MtfCurve:
    """Calculate the MTF from an image of an edge phantom. Find the edge using a canny filter in combination with the
    Hough transform. The raw ESF is smoothed using a Savitzky-Golay filter with a 4th order polynomial.

    Args:
        image: A numpy ndarray of the image containing the edge phantom
        roi: A square ROI object specifying the region in which the edge phantom is contained
        acceptable_line_gap: The maximum acceptable number of pixels that form a gap in the line that the probabilitic Hough function is trying to find

    Returns:
        An :obj:`MtfCurve <dicom_image_tools.image_quality_mtf.MtfCurve>` object instance with the frequency in
        cycles/pixel

    """
    if not isinstance(acceptable_line_gap, int):
        raise TypeError("The acceptable line gap must be given as an integer")

    # Find edge
    edge_image = roi.get_roi_part_of_image(image=image)
    blurred_edge_image = gaussian(edge_image, sigma=5)
    # Add a rescale intensity to increase the probability that the canny filter finds the edge
    blurred_edge_image = rescale_intensity(
        blurred_edge_image, in_range=(blurred_edge_image.min(), blurred_edge_image.max())
    )
    edges = canny(
        image=blurred_edge_image,
        sigma=1,
        low_threshold=blurred_edge_image.min(),
        high_threshold=blurred_edge_image.min() + (blurred_edge_image.max() - blurred_edge_image.min()) / 2,
    )

    lines = probabilistic_hough_line(
        image=edges, line_gap=acceptable_line_gap, line_length=round(min(edge_image.shape[:2]) * 0.95)
    )
    line = Line(x0=lines[0][0][0], y0=lines[0][0][1], x1=lines[0][1][0], y1=lines[0][1][1])

    xx, yy = np.meshgrid(np.arange(edge_image.shape[1]), np.arange(edge_image.shape[0]))
    point_data = [
        (get_shortest_line_point_dist(line=line, point=IntPoint(x=obj[0], y=obj[1])), obj[2])
        for obj in np.vstack((xx.ravel(), yy.ravel(), edge_image.ravel())).T
    ]
    point_data = sorted(point_data, key=lambda p: p[0])
    x, y = zip(*point_data)

    # Smooth the raw edge
    window_length = int(np.floor(np.divide(len(y), 100)))
    if window_length % 2 == 0:
        window_length += 1
    y_smooth = savgol_filter(x=y, window_length=window_length, polyorder=4)

    # Make sure that the edge has a higher value for positive x-values
    if np.mean(y_smooth[0 : int(len(y_smooth) / 100)]) > np.mean(y_smooth[-int(len(y_smooth) / 100) :]):
        # y = np.array(list(reversed(y)))
        y_smooth = np.array(list(reversed(y_smooth)))
        x = np.array([-el for el in reversed(x)])

    selection = np.where((x > -20) & (x < 20))
    y_smooth = y_smooth[selection]
    # y = y[selection]
    x = x[selection]

    dy = np.diff(y_smooth)
    # dx = np.diff(x)

    mean, std = norm.fit(dy)
    y_fit = norm.pdf(x, mean, std)

    mtf = abs(np.fft.rfft(y_fit))
    mtf = mtf / max(mtf)

    return MtfCurve(frequency=[i for i in range(len(mtf))], mtf=mtf)
