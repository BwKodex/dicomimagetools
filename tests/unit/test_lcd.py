import numpy as np
import pytest

from dicom_image_tools import SquareRoi
from dicom_image_tools.helpers.voxel_data import VoxelData
from dicom_image_tools.image_quality.lcd import lcd_statistical, lcd_statistical_random

TEST_MATRIX = np.asarray(
    [
        [2, -7, 16, -3, -1, -15, 12, 5, 19, -2, -3, 13, -6, 19, 9, 3, -14, 12, 14, -11],
        [-19, -17, -2, 10, 13, 7, -13, 3, 0, 1, -10, -13, -1, -10, -8, 14, 11, 12, 18, 12],
        [-4, 17, 1, -9, 19, 0, -15, -16, -19, -9, 19, 16, 16, 16, 1, 16, 1, 11, -13, 3],
        [-7, -13, -18, -5, 8, 14, 0, 4, 16, -9, -9, 14, 4, -9, 17, -20, 2, 6, -17, 13],
        [-13, 2, 17, -18, -11, -13, -18, 17, 9, -8, -6, -13, -18, -16, 12, -3, 1, -18, -17, 5],
        [17, -15, -13, -13, -9, 15, 12, -8, 3, 1, -14, -6, -3, -15, -18, 9, -5, -2, 15, 1],
        [-11, -18, -13, -16, -12, 11, 17, -18, -19, 7, 6, -1, -15, 18, -11, -1, 17, -2, 3, 8],
        [15, -13, 14, 10, 12, 10, 13, -15, 12, -20, 14, 6, -19, 10, 11, -14, -1, 2, 16, -19],
        [-4, -4, -9, 11, -15, -4, -5, 18, 13, -15, 15, 2, 2, -11, -2, 7, 9, -20, -12, 2],
        [11, -9, 10, -14, -9, 17, -20, 3, -17, -8, 17, 5, 1, -19, -6, -7, 13, -12, -8, -17],
        [1, 19, 6, -12, 12, 0, -8, -5, -5, 10, 13, 9, -7, -13, 19, -3, 13, -15, -11, 1],
        [-3, 6, -4, 11, -12, 10, 5, 13, 15, 14, -8, 5, 3, 19, 16, 6, 9, -5, -9, 1],
        [-18, 3, 11, -14, 8, -18, 8, -6, -4, -20, 17, -16, 19, -1, 8, -17, -20, 12, 11, -12],
        [12, -14, 13, 16, -3, 1, -13, 3, -18, -9, -3, -4, -9, 18, -14, 15, -1, -19, -19, 18],
        [4, -2, -18, 1, -3, -9, 12, -8, 11, 12, 14, -1, 10, -2, 7, -19, 0, 10, 17, 8],
        [11, 3, -16, 6, 11, -14, -8, -17, -11, -5, 2, -9, -11, -12, -2, -20, -7, 2, -9, -19],
        [-15, 17, 13, -8, -6, 16, 7, -11, 1, -13, 18, 17, 18, 18, -7, -10, 1, 3, -11, 19],
        [-1, 7, 4, 7, -3, -5, -4, -4, 9, 1, 18, -8, -2, 16, 11, -19, -6, -6, 2, -7],
        [4, -17, -2, -15, -4, -11, -9, 5, 7, -11, -3, 1, -14, -8, -12, 17, 3, -6, -2, 11],
        [-13, 8, -1, -14, -8, 2, 3, 3, 6, 19, 9, 3, -19, -1, -17, -7, 17, 4, 12, -5],
    ]
)


def test_lcd_statistical_raises_stderr_typeerror():
    with pytest.raises(TypeError):
        lcd_statistical(stderr="Invalid argument type")


def test_lcd_statistical_return_float():
    test_val = 4.0
    expected = 3.29 * test_val

    test = lcd_statistical(stderr=test_val)

    assert isinstance(test, float)
    assert test == expected


def test_lcd_statistical_random_raises_analysis_matrix_typeerror():
    with pytest.raises(TypeError):
        lcd_statistical_random(
            analysis_matrix="Invalid argument type", pixel_size=VoxelData(x=1.0, y=1.0), object_size=1.0
        )


def test_lcd_statistical_random_raises_pixel_size_typeerror():
    with pytest.raises(TypeError):
        lcd_statistical_random(analysis_matrix=TEST_MATRIX, pixel_size="Invalid argument type", object_size=1.0)


def test_lcd_statistical_random_raises_object_size_typeerror():
    with pytest.raises(TypeError):
        lcd_statistical_random(
            analysis_matrix=TEST_MATRIX, pixel_size=VoxelData(x=1.0, y=1.0), object_size="Invalid argument type"
        )


def test_lcd_statistical_random_raises_rois_typeerror():
    with pytest.raises(TypeError):
        lcd_statistical_random(
            analysis_matrix=TEST_MATRIX,
            pixel_size=VoxelData(x=1.0, y=1.0),
            object_size=1.0,
            rois="Invalid argument type",
        )


def test_lcd_statistical_random():
    res = lcd_statistical_random(analysis_matrix=TEST_MATRIX, pixel_size=VoxelData(x=1.0, y=1.0), object_size=3.0)
    assert res.get("Hole Diameter") == "3.00mm"
    assert res.get("ROI Box Size") == "3pixel"
    assert np.min(TEST_MATRIX) < res.get("Mean") < np.max(TEST_MATRIX)
    assert not np.isnan(res.get("Std Error Mean"))
    assert not np.isnan(res.get("SD"))
    assert not np.isnan(res.get("Error SD"))
    assert not np.isnan(res.get("Perc Contrast At 95 Perc CL"))
    assert res.get("ROIs") is None


def test_lcd_statistical_random_returns_rois():
    expected_rois = 1500

    res = lcd_statistical_random(
        analysis_matrix=TEST_MATRIX,
        pixel_size=VoxelData(x=1.0, y=1.0),
        object_size=3.0,
        return_rois=True,
        rois=expected_rois,
    )

    rois = res.get("ROIs")

    assert rois is not None
    assert len(rois) == expected_rois
    assert all([isinstance(roi, SquareRoi) for roi in rois])
