import numpy as np
import pytest

from dicom_image_tools.helpers.remove_position_indication_square import (
    remove_position_indication_square_schick_sensor_image,
)


def test_remove_position_indication_square_schick_sensor_image_removes_square_for_sirona_sensor(io_series_sirona):
    # Arrange
    expected = 3780
    io_series_sirona.import_image(rotate_to_0_degerees=True)

    # Act
    result = remove_position_indication_square_schick_sensor_image(
        image=io_series_sirona.ImageVolume[0], metadata=io_series_sirona.CompleteMetadata[0]
    )
    actual = int(np.ceil(np.mean(result[10:23, 10:23])))

    # Assert
    assert actual == expected


def test_remove_position_indication_square_schick_sensor_image_raises_NotImplementedError_when_detecotr_manufacturer_not_in_metadata(
    io_series_sirona,
):
    # Arrange
    io_series_sirona.import_image(rotate_to_0_degerees=True)
    del io_series_sirona.CompleteMetadata[0].DetectorManufacturerName

    # Assert
    with pytest.raises(NotImplementedError):
        _ = remove_position_indication_square_schick_sensor_image(
            image=io_series_sirona.ImageVolume[0], metadata=io_series_sirona.CompleteMetadata[0]
        )


def test_remove_position_indication_square_schick_sensor_image_raises_NotImplementedError_when_detecotr_from_unsupported_manufacturer(
    io_series_sirona,
):
    # Arrange
    io_series_sirona.import_image(rotate_to_0_degerees=True)
    io_series_sirona.CompleteMetadata[0].DetectorManufacturerName = "Unsupported Manufacturer"

    # Assert
    with pytest.raises(NotImplementedError):
        _ = remove_position_indication_square_schick_sensor_image(
            image=io_series_sirona.ImageVolume[0], metadata=io_series_sirona.CompleteMetadata[0]
        )
