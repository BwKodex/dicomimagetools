import numpy as np
from pydicom import FileDataset

from dicom_image_tools.helpers.window import get_default_window_settings


def test_get_default_window_setting_returns_plus_minus_500_when_ct_has_no_window_tag_data(
    image_with_negative_pixel_intensity_relationship,
):
    # Arrange
    metadata = image_with_negative_pixel_intensity_relationship.get("metadata")
    image_array = np.array([1, 2, 3])
    expected = (-500, 500)

    # Act
    actual = get_default_window_settings(metadata=metadata, image_slice=image_array, modality="CT")

    # Assert
    assert actual == expected


def test_get_default_window_setting_returns_correct_window_for_ct_with_windowing_metadata(
    image_with_negative_pixel_intensity_relationship,
):
    # Arrange
    expected = (-234, 236)
    metadata: FileDataset = image_with_negative_pixel_intensity_relationship.get("metadata")
    window_width = expected[1] - expected[0]
    metadata.WindowWidth = window_width
    metadata.WindowCenter = expected[0] + window_width / 2
    image_array = np.array([1, 2, 3])

    # Act
    actual = get_default_window_settings(metadata=metadata, image_slice=image_array, modality="CT")

    # Assert
    assert actual == expected


def test_get_default_window_setting_returns_max_min_window_for_non_ct_modality_without_windowing_metadata(
    image_with_negative_pixel_intensity_relationship,
):
    # Arrange
    expected = (-12, 19)
    metadata: FileDataset = image_with_negative_pixel_intensity_relationship.get("metadata")
    image_array = np.array([expected[1], expected[0], 3])

    # Act
    actual = get_default_window_settings(metadata=metadata, image_slice=image_array, modality="PX")

    # Assert
    assert actual == expected
