import numpy as np
import pytest

from dicom_image_tools.helpers.rotate_image import rotate_image


def test_rotate_image_correctly_rotates_image(io_series_sirona):
    # Arrange
    expected_before_rotation = 3765
    expected_after_rotation = 317
    io_series_sirona.import_image()
    image = io_series_sirona.ImageVolume[0]

    # Act
    actual_before_rotation = int(np.ceil(np.mean(image[10:23, 10:23])))
    actual_after_rotation = int(
        np.ceil(np.mean(rotate_image(image=image, metadata=io_series_sirona.CompleteMetadata[0])[10:23, 10:23]))
    )

    # Assert
    assert actual_before_rotation == expected_before_rotation
    assert actual_after_rotation == expected_after_rotation


def test_rotate_image_riases_ValueError_if_FieldOfViewRotation_not_in_metadata(io_series_sirona):
    # Arrange
    expected_message = "No field of view rotation data in the given metadata"
    io_series_sirona.import_image()
    del io_series_sirona.CompleteMetadata[0].FieldOfViewRotation

    # Assert
    with pytest.raises(ValueError) as exc:
        rotate_image(image=io_series_sirona.ImageVolume[0], metadata=io_series_sirona.CompleteMetadata[0])

    assert expected_message == str(exc.value)
