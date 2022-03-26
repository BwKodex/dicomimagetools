import pytest


def test_projection_series_get_variance_images_for_image_volume_raises_value_error_when_image_volume_not_imported(
    io_series_sirona,
):
    # Arrange
    io_series_sirona.ImageVolume = []

    # Act/Assert
    with pytest.raises(ValueError) as exc:
        io_series_sirona.get_variance_images_for_image_volume()

    assert "The image must be imported before the variance images can be calculated" in str(exc.value)


def test_projection_series_get_variance_images_for_image_volume_returns_list_of_ndarrays_of_same_shape_as_images(
    io_series_sirona,
):
    # Arrange
    io_series_sirona.import_image()
    expected_length = len(io_series_sirona.ImageVolume)
    expected_shapes = [im.shape for im in io_series_sirona.ImageVolume]

    # Act
    result = io_series_sirona.get_variance_images_for_image_volume()
    actual_length = len(result)
    actual_shapes = [im.shape for im in result]

    assert actual_length == expected_length
    assert actual_shapes == expected_shapes
