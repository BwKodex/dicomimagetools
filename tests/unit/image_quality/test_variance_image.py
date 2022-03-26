import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from dicom_image_tools.image_quality.variance_image import get_variance_image_2d

TEST_MATRIX = np.asarray(
    [
        [10, 12, 11, 15, 19],
        [13, 16, 14, 8, 12],
        [12, 15, 18, 10, 11],
        [7, 14, 9, 15, 18],
        [11, 15, 9, 13, 7],
    ]
)


def test_get_variance_image_2d_correctly_calculates_the_variance_image():
    # Arrange
    expected = np.asarray(
        [
            [3.728, 3.432, 5.777, 12.172, 14.444],
            [3.580, 5.802, 9.061, 12.098, 13.333],
            [8.987, 10.320, 10.395, 12.172, 10.839],
            [8.024, 11.283, 8.765, 14.395, 15.283],
            [8.000, 7.654, 6.469, 12.987, 20.222],
        ]
    )

    # Act
    actual = get_variance_image_2d(TEST_MATRIX, window_side_x=3, window_side_y=3)

    # Assert
    assert_array_almost_equal(actual, expected, decimal=3)


def test_get_variance_image_2d_raises_type_error_when_image_is_not_an_ndarray():
    # Assert
    with pytest.raises(TypeError) as exc:
        get_variance_image_2d(image=12)

    assert str(exc.value).startswith("The image must be a numpy ndarray")


def test_get_variance_image_2d_raises_type_error_when_window_side_x_is_not_int():
    # Assert
    with pytest.raises(TypeError) as exc:
        # noinspection PyTypeChecker
        get_variance_image_2d(image=TEST_MATRIX, window_side_x="invalid")

    assert str(exc.value).startswith("The window sides have to be specified as integers.")


def test_get_variance_image_2d_raises_type_error_when_window_side_y_is_not_int():
    # Assert
    with pytest.raises(TypeError) as exc:
        # noinspection PyTypeChecker
        get_variance_image_2d(image=TEST_MATRIX, window_side_y="invalid")

    assert str(exc.value).startswith("The window sides have to be specified as integers.")
