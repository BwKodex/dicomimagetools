import numpy as np
import pytest

from dicom_image_tools.helpers.voxel_data import VoxelData
from dicom_image_tools.roi.square_roi import SquareRoi


test_image = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
        [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    ]
)


def test_too_small_roi():
    center = dict(x=6, y=6, z=None)
    voxel_data = VoxelData(x=10.0, y=10.0, z=None)
    with pytest.raises(ValueError):
        SquareRoi(center=center, height=1, width=1, pixel_size=voxel_data)


def test_square_roi_creation():
    center = dict(x=6, y=6, z=None)
    voxel_data = VoxelData(x=0.5, y=0.5, z=None)
    square_roi = SquareRoi(center=center, height=3, width=3, pixel_size=voxel_data)
    assert square_roi.Center.x == 6
    assert square_roi.Center.y == 6
    assert square_roi.Center.z is None
    assert square_roi.Height == 3
    assert square_roi.Width == 3
    assert square_roi.UpperLeft.x == 4
    assert square_roi.UpperLeft.y == 4
    assert square_roi.LowerLeft.x == 4
    assert square_roi.LowerLeft.y == 9
    assert square_roi.UpperRight.x == 9
    assert square_roi.UpperRight.y == 4
    assert square_roi.LowerRight.x == 9
    assert square_roi.LowerRight.y == 9


def test_unallowed_color():
    center = dict( x=6, y=6, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    square_roi = SquareRoi(center=center, height=3, width=3, pixel_size=voxel_data)
    with pytest.raises(ValueError):
        square_roi.add_roi_to_image(image=test_image, roi_color="Not a real colour")
