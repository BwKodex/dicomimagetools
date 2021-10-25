import logging

import numpy as np
from pydicom import Dataset

log = logging.getLogger(__name__)


def get_pixel_array(dcm: Dataset) -> np.ndarray:
    """Take a DICOM dataset, extract the pixels, rescale the values according to metadata. Return the extracted image
    as a numpy ndarray of int16 values

    Args:
        dcm: The DICOM dataset from which the image should be extracted

    Returns:
        Extracted image as a numpy ndarray of int16 numbers

    """
    px = dcm.pixel_array.astype(np.float)

    if "Modality" in dcm and dcm.Modality.casefold() == "RTDOSE".casefold():
        return rescale_dose_matrix_pixel_array(pixel_array=px, dcm=dcm)

    if "RescaleSlope" in dcm:
        log.debug("Rescaling slope of pixel array")
        px *= np.float(dcm.RescaleSlope)

    if "RescaleIntercept" in dcm:
        log.debug("Rescaling intercept of pixel array")
        px += np.float(dcm.RescaleIntercept)

    return px


def rescale_dose_matrix_pixel_array(pixel_array: np.ndarray, dcm: Dataset) -> np.ndarray:
    """ Rescaled the dose matrix by the DoseGridScaling tag value if it is present in the dataset

    Args:
        pixel_array: The dose matrix as a numpy.ndarray
        dcm: The dataset containing information on the dose matrix

    Returns:
        The rescaled dose matrix as a numpy.ndarray of int16 numbers

    """
    if "DoseGridScaling" in dcm:
        scaling_factor = dcm.DoseGridScaling
        log.debug(f"Rescaling dose matrix by DoseGridScaling of {scaling_factor}")
        pixel_array = pixel_array.astype(np.float) * np.float(scaling_factor)

    return pixel_array
