import numpy as np
from typing import List


def mtf_bar_pattern(max_hu: float, bg: float, sd: List[float], sd_bg: float) -> List[float]:
    """ Implement the method by Droege and Morin (1982) Medical Physics. Determine MTF from bar pattern data

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
