import numpy as np
from scipy import ndimage


def get_variance_image_2d(image: np.ndarray, window_side_x: int = 3, window_side_y: int = 3) -> np.ndarray:
    """Calculates and returns a variance image using a rolling window with the side lengths specified. Returns a
    variance image of the same dimensions as the input image

    Args:
        image: An image for which the variance image should be calculated
        window_side_x: The horizontal side (columns) of the rolling window applied. Default = 3
        window_side_y: The vertical side (rows) of the rolling window applied. Default = 3

    Returns:
        A numpy ndarray of the same dimensions as the input image showing the variance at each pixel
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"The image must be a numpy ndarray, was given as {type(image)}")
    if not isinstance(window_side_x, int) or not isinstance(window_side_y, int):
        raise TypeError(
            f"The window sides have to be specified as integers. Current types: window_side_x={type(window_side_x)}, "
            f"window_side_y={type(window_side_y)}"
        )

    return ndimage.generic_filter(image.astype(float), np.var, size=(window_side_y, window_side_x))
