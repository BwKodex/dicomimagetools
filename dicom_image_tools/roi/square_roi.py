import numpy as np
from scipy.stats import sem
from skimage import color

from .roi import Roi, CenterPosition
from ..helpers.point import IntPoint
from ..helpers.voxel_data import VoxelData


VALID_COLOURS = {
    'SkyBlue': (135, 206, 235),
    'MediumBlue': (0, 0, 205)
}


class SquareRoi(Roi):
    """
    A class for square ROIs in images.

    Attributes
    ----------
    center : CenterPosition
        The coordinates for the center position (x-, y-, and z-index) of the ROI
    height : float
        ROI height in mm
    width : float
        ROI width in mm
    pixel_size : VoxelData
        Pixel size/spacing in mm

    Methods
    -------
    get_mean(image)
        Returns the mean of the pixel values in the ROI
    get_stdev(image)
        Returns the standard deviation of the pixel values in the ROI
    get_sum(image, axis=1)
        Returns the sum of the pixel values in the ROI
    get_std_error_of_the_mean(image)
        Returns the standard error of the mean of the pixel values in the ROI
    add_roi_to_image(image, roi_color='SkyBlue')
        Returns the image with the ROI added, coloured in the requested colour
    get_roi_part_of_image(image)
        Returns a numpy ndarray only containing the part of the image that is contained in the ROI
    """

    def __init__(self, center: CenterPosition, height: float, width: float, pixel_size: VoxelData):
        super().__init__(center=center)
        self.Height: float = height
        self.Width: float = width

        height_pixels = int(round(self.Height / pixel_size.y))
        width_pixels = int(round(self.Width / pixel_size.x))

        if height_pixels < 1 or width_pixels < 1:
            raise ValueError((f'Too small ROI size specified. Both width ({width_pixels}) and height ({height_pixels}) '
                              f'must be at least 1 pixel'))

        self.UpperLeft: IntPoint = IntPoint(
            x=int(round(self.Center.x - (width_pixels / 2))),
            y=int(round(self.Center.y - (height_pixels / 2)))
        )
        self.LowerLeft: IntPoint = IntPoint(
            x=int(round(self.Center.x - (width_pixels / 2))),
            y=int(round(self.Center.y + (height_pixels / 2)))
        )
        self.UpperRight: IntPoint = IntPoint(
            x=int(round(self.Center.x + (width_pixels / 2))),
            y=int(round(self.Center.y - (height_pixels / 2)))
        )
        self.LowerRight: IntPoint = IntPoint(
            x=int(round(self.Center.x + (width_pixels / 2))),
            y=int(round(self.Center.y + (height_pixels / 2)))
        )

    def get_mean(self, image: np.ndarray) -> np.ndarray:
        """ Calculates the mean of the pixel values contained in the ROI from the input image

        :param image: Numpy ndarray containing the image
        :return: The mean of the pixel values from the part of the input image contained in the ROI
        """
        return np.mean(image[self.UpperLeft.y:self.LowerRight.y + 1, self.UpperLeft.x:self.LowerRight.x + 1])

    def get_stdev(self, image: np.ndarray) -> np.ndarray:
        """ Calculates the standard deviation of the pixel values contained in the ROI from the input image

        :param image: Numpy ndarray containing the image
        :return: The standard deviation of the pixel values from the part of the input image contained in the ROI
        """
        return np.std(image[self.UpperLeft.y:self.LowerRight.y + 1, self.UpperLeft.x:self.LowerRight.x + 1])

    def get_sum(self, image: np.ndarray, axis: int = 1):
        """ Calculates the sum of the pixel values contained in the ROI from the input image

        :param image: Numpy ndarray containing the image
        :param axis: The axis over which to calculate the sum
        :return: The sum of the square ROI applied to the supplied image over the specified axis
        """
        return np.sum(a=image[self.UpperLeft.y:self.LowerRight.y + 1, self.UpperLeft.x:self.LowerRight.x + 1],
                      axis=axis)

    def get_std_error_of_the_mean(self, image: np.ndarray) -> np.ndarray:
        """ Calculates the standard error of the mean of the pixel values contained in the ROI from the input image

        :param image: Numpy ndarray containing the image
        :return: The standard error of the mean of the pixel values from the part of the input image contained in the
        ROI
        """
        return sem(image[self.UpperLeft.y:self.LowerRight.y + 1, self.UpperLeft.x, self.LowerRight.x + 1].flatten())

    def add_roi_to_image(self, image: np.ndarray, roi_color: str = 'SkyBlue') -> np.ndarray:
        """ Adds the ROI to the given image using the color specified.

        :param image: Numpy ndarray containing the image
        :param roi_color: The color to use for the ROI
        :return: The input image with the ROI added
        """
        roi_color = VALID_COLOURS.get(roi_color)

        if roi_color is None:
            raise ValueError(f'The colour must be one of {",".join(VALID_COLOURS.keys())}')

        image = color.gray2rgb(image)

        image[self.UpperLeft.y:self.LowerRight.y + 1, self.UpperLeft.x:self.LowerRight.x + 1] = roi_color
        return image

    def get_roi_part_of_image(self, image: np.ndarray) -> np.ndarray:
        """ Extracts the part of the image contained in the ROI

        :param image: Numpy ndarray containing the image
        :return: The part of the input image contained in the square ROI as a numpy ndarray
        """
        return image[self.UpperLeft.y:self.LowerRight.y + 1, self.UpperLeft.x:self.LowerRight.x + 1]
