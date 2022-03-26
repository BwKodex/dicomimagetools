import tempfile

import numpy as np
from pydicom import Dataset

from dicom_image_tools.helpers.pixel_data import (
    get_pixel_array,
    rescale_dose_matrix_pixel_array,
)


def test_rescale_dose_matrix_pixel_array_rescales_the_pixel_array_by_the_dose_grid_scaling_factor(dose_matrix):
    # Arrange
    scaling_factor = 10.0
    expected = max(dose_matrix.flatten()) * scaling_factor
    ds: Dataset = Dataset()
    ds.DoseGridScaling = scaling_factor

    # Act
    actual = max(rescale_dose_matrix_pixel_array(pixel_array=dose_matrix, dcm=ds).flatten())

    # Assert
    assert actual == expected


def test_rescale_dose_matrix_pixel_array_returns_same_object_if_dose_grid_scaling_not_in_dataset(dose_matrix):
    # Arrange
    expected = dose_matrix.copy()

    # Act
    actual = rescale_dose_matrix_pixel_array(pixel_array=expected, dcm=Dataset())

    # Assert
    assert np.testing.assert_array_equal(actual, expected) is None


def test_get_pixel_array_rescales_by_DoseGridScaling_when_rtdose_modality(dose_matrix, dose_matrix_file):
    # Arrange
    scaling_factor = 0.12
    expected = max(dose_matrix.flatten()) * scaling_factor

    ds = dose_matrix_file.copy()
    ds.DoseGridScaling = scaling_factor
    ds.BitsStored = 12

    # Act
    actual = max(get_pixel_array(dcm=ds).flatten())

    # Assert
    assert actual == expected
