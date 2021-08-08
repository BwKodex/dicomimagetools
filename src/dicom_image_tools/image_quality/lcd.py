import logging
from typing import Dict, List, Optional, Union

import numpy as np
from scipy.stats import sem

from ..helpers.point import Point
from ..helpers.voxel_data import VoxelData
from ..roi.square_roi import SquareRoi

FloatListType = Union[float, List[float]]

LCD_CONSTANT = 3.29  # Value from A. Radice et al. (2016) Physica Medica (32)

log = logging.getLogger(__name__)


def lcd_statistical(stderr: FloatListType) -> FloatListType:
    """Implements a statistical method for determining the low contrast detectability (LCD) as mentioned in
    "A STATISTICAL METHOD FOR LOW-CONTRAST DETECTABILITY ANALYSIS IN ANGIOGRAPHY SYSTEMS. A. Radice, N. Paruccini,
    C. Spadavecchia, R. Villa, A. Baglivi and A. Crespi (2016) Physica Medica (32).
    https://doi.org/10.1016/j.ejmp.2016.07.728" This specifies that the low contrast detectability as 3.29 x sigma,
    where sigma is the standard error of the mean of 100 square ROIS of a side corresponding to the "hole" size that the
    LCD is determined for.

    Args:
        stderr: Standard error of 100 ROIs given as a single value or a list of the standard error of each ROI

    Returns:
        Low contrast resolution as a float (if stderr is a float) or a list of floats (if stderr is a list of floats)
    """
    if not isinstance(stderr, float) and not isinstance(stderr, list):
        raise TypeError("stderr must be given as a float or a list of floats")

    if isinstance(stderr, float):
        return LCD_CONSTANT * stderr

    return [LCD_CONSTANT * val for val in stderr]


def lcd_statistical_random(
    analysis_matrix: np.ndarray,
    pixel_size: VoxelData,
    object_size: float,
    rois: Optional[int] = 1000,
    return_rois: Optional[bool] = False,
) -> Dict[str, Union[str, float, List[float]]]:
    """Implements a statistical method for determining the low contrast detectability (LCD) or low contrast limit (LCL)

    Args:
        analysis_matrix: Matrix of the homogeneous image area that is to be analysed
        pixel_size: The voxel data for the analysis_matrix
        object_size: The side of the object to analyse for in mm
        rois: The number of ROIs to include in the analysis

    Returns:
        Dictionary containing the mean, standard deviation, standard error and LCD on the form
        {
          'Hole Diameter': str,                  # (mm)
          'ROI Box Size': str,                   # (pixels)
          'Mean': float,                         # Mean value
          'Std Error Mean': float,               # Standard error of the mean
          'SD': float,                           # Standard deviation
          'Error SD': float,                     # Standard error of the standard deviation
          'Perc contrast At 95 Perc CL': float,  # Percent contrast at 95% contrast level
          'ROIs': Optional[List[SquareRoi]],     # A list of all ROIs used or None
        }

    """
    if not isinstance(analysis_matrix, np.ndarray):
        raise TypeError("The analysis matrix must be a numpy ndarray")

    if not isinstance(pixel_size, VoxelData):
        raise TypeError("The pixel_size must be given as a VoxelData object")

    if not isinstance(object_size, float):
        raise TypeError("Object size must be a float")

    if not isinstance(rois, int):
        raise TypeError("The number of ROIs (rois) must be an integer")

    output = {
        "Hole Diameter": f"{object_size:.2f}mm",
        "ROI Box Size": f"{int(np.floor(object_size / pixel_size.x))}pixel",
        "Mean": float(),
        "Std Error Mean": float(),
        "SD": float(),
        "Error SD": float(),
        "Perc Contrast At 95 Perc CL": float(),
    }

    result = {
        "Mean": [],
        "SD": [],
        "Std Error Mean": [],
        "ROIs": [],
    }

    log.info(f"Calculating LCD from a set of {rois} ROIs of size {object_size:.3f} mm")

    # Place ROIs and get ROI statistics
    roi_size = [np.floor(np.divide(object_size, pixel_size.x)), np.floor(np.divide(object_size, pixel_size.y))]
    structured_rois = [analysis_matrix.shape[1] // roi_size[0], analysis_matrix.shape[0] // roi_size[1]]
    random_rois = int(np.floor(rois - structured_rois[0] * structured_rois[1]))

    # Determine the mean values and standard deviation for a structured ROI set
    for col in range(np.int(structured_rois[0])):
        for row in range(np.int(structured_rois[1])):
            roi = SquareRoi(
                center=Point(
                    x=int(col * roi_size[0] + np.ceil(roi_size[0] / 2)),
                    y=int(row * roi_size[1] + np.ceil(roi_size[1] / 2)),
                ),
                height=object_size,
                width=object_size,
                pixel_size=pixel_size,
                resize_too_big_roi=True,
            )
            result["Mean"].append(roi.get_mean(analysis_matrix))
            result["SD"].append(roi.get_stdev(analysis_matrix))
            result["Std Error Mean"].append(roi.get_std_error_of_the_mean(analysis_matrix))
            result["ROIs"].append(roi)

    if random_rois > 0:
        # Determine the mean value and standard deviation in a set of randomly placed ROIs
        for _ in range(random_rois):
            valid_roi = False
            n = 0
            while not valid_roi and n < 1000:
                roi = SquareRoi(
                    center=Point(
                        x=int(np.random.randint(0, analysis_matrix.shape[1]) + np.ceil(np.divide(roi_size[0], 2))),
                        y=int(np.random.randint(0, analysis_matrix.shape[0]) + np.ceil(np.divide(roi_size[1], 2))),
                    ),
                    height=object_size,
                    width=object_size,
                    pixel_size=pixel_size,
                    resize_too_big_roi=True,
                )
                tmp_mean = roi.get_mean(analysis_matrix)
                tmp_sd = roi.get_stdev(analysis_matrix)
                tmp_sem = roi.get_std_error_of_the_mean(analysis_matrix)

                n += 1
                valid_roi = not np.isnan(tmp_mean) and not np.isnan(tmp_sd) and not np.isnan(tmp_sem)

            if valid_roi:
                result["Mean"].append(tmp_mean)
                result["SD"].append(tmp_sd)
                result["Std Error Mean"].append(tmp_sem)
                result["ROIs"].append(roi)

    # Calculate the mean value and related values from the ROIs
    output["Mean"] = np.mean(result["Mean"])
    output["Std Error Mean"] = np.mean(result["Std Error Mean"])
    output["SD"] = np.mean(result["SD"])
    output["Error SD"] = sem(result["SD"])
    output["Perc Contrast At 95 Perc CL"] = lcd_statistical(stderr=output["Std Error Mean"])

    if return_rois:
        output["ROIs"] = result["ROIs"]

    return output
