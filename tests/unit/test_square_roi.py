import numpy as np
from scipy.stats import sem
from skimage import color
import pytest

from dicom_image_tools.helpers.voxel_data import VoxelData
from dicom_image_tools.roi.square_roi import SquareRoi, VALID_COLOURS


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


def test_too_small_roi():
    center = dict(x=6, y=6, z=None)
    voxel_data = VoxelData(x=10.0, y=10.0, z=None)
    with pytest.raises(ValueError):
        SquareRoi(center=center, height=1, width=1, pixel_size=voxel_data)


def test_upper_left_outside():
    center = dict(x=1, y=1, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    with pytest.raises(ValueError) as exc_info:
        SquareRoi(center=center, height=5, width=5, pixel_size=voxel_data)
    assert 'upper left corner' in str(exc_info.value)


def test_upper_left_outside_resize():
    center = dict(x=1, y=1, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    square_roi = SquareRoi(center=center, height=5, width=5, pixel_size=voxel_data, resize_too_big_roi=True)
    assert square_roi.UpperLeft.x == 0
    assert square_roi.UpperLeft.y == 0


def test_check_roi_placement_outside():
    center = dict(x=100, y=100, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    square_roi = SquareRoi(center=center, height=3, width=3, pixel_size=voxel_data)
    with pytest.raises(ValueError):
        square_roi._check_roi_placement(image=test_image)


def test_check_roi_placement_part_outside():
    center = dict(x=12, y=12, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    square_roi = SquareRoi(center=center, height=3, width=3, pixel_size=voxel_data)
    with pytest.raises(ValueError):
        square_roi._check_roi_placement(image=test_image)


def test_get_mean():
    expected_mean = 7.0

    center = dict(x=6, y=6, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    square_roi = SquareRoi(center=center, height=3, width=3, pixel_size=voxel_data)

    assert expected_mean == square_roi.get_mean(image=test_image)


def test_get_mean_too_big_roi_resize():
    expected_mean = 12.0

    center = dict(x=12, y=12, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    square_roi = SquareRoi(center=center, height=3, width=3, pixel_size=voxel_data, resize_too_big_roi=True)

    assert expected_mean == square_roi.get_mean(image=test_image)


def test_get_mean_too_big_roi_no_resize():
    center = dict(x=12, y=12, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    square_roi = SquareRoi(center=center, height=3, width=3, pixel_size=voxel_data, resize_too_big_roi=False)

    with pytest.raises(ValueError):
        square_roi.get_mean(image=test_image)


def test_get_stdev():
    expected_stdev = np.std(test_image[5:7 + 1, 5:7 + 1])

    center = dict(x=6, y=6, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    square_roi = SquareRoi(center=center, height=3, width=3, pixel_size=voxel_data)

    assert expected_stdev == square_roi.get_stdev(image=test_image)


def test_get_stdev_too_big_roi_resize():
    expected_stdev = 0.5

    center = dict(x=12, y=12, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    square_roi = SquareRoi(center=center, height=5, width=5, pixel_size=voxel_data, resize_too_big_roi=True)

    assert expected_stdev == square_roi.get_stdev(image=test_image)


def test_get_stdev_too_big_roi_no_resize():
    center = dict(x=12, y=12, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    square_roi = SquareRoi(center=center, height=3, width=3, pixel_size=voxel_data, resize_too_big_roi=False)

    with pytest.raises(ValueError):
        square_roi.get_stdev(image=test_image)


def test_get_sum():
    expected_stdev = sum(np.sum(test_image[5:7 + 1, 5:7 + 1], axis=1))

    center = dict(x=6, y=6, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    square_roi = SquareRoi(center=center, height=3, width=3, pixel_size=voxel_data)

    assert expected_stdev == sum(square_roi.get_sum(image=test_image))


def test_get_sum_too_big_roi_resize():
    expected = 46

    center = dict(x=12, y=12, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    square_roi = SquareRoi(center=center, height=5, width=5, pixel_size=voxel_data, resize_too_big_roi=True)

    assert expected == sum(square_roi.get_sum(image=test_image))


def test_get_sum_too_big_roi_no_resize():
    center = dict(x=12, y=12, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    square_roi = SquareRoi(center=center, height=3, width=3, pixel_size=voxel_data, resize_too_big_roi=False)

    with pytest.raises(ValueError):
        square_roi.get_sum(image=test_image)


def test_get_standard_error_of_them_mean():
    expected = sem(test_image[5:7 + 1, 5:7 + 1].flatten())

    center = dict(x=6, y=6, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    square_roi = SquareRoi(center=center, height=3, width=3, pixel_size=voxel_data)

    assert expected == square_roi.get_std_error_of_the_mean(image=test_image)


def test_get_standard_error_of_them_mean_too_big_roi_resize():
    expected = sem(test_image[10:12 + 1, 10:12 + 1].flatten())

    center = dict(x=12, y=12, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    square_roi = SquareRoi(center=center, height=5, width=5, pixel_size=voxel_data, resize_too_big_roi=True)

    assert square_roi.get_std_error_of_the_mean(image=test_image) == expected


def test_get_standard_error_of_them_mean_too_big_roi_no_resize():
    center = dict(x=12, y=12, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    square_roi = SquareRoi(center=center, height=3, width=3, pixel_size=voxel_data, resize_too_big_roi=False)

    with pytest.raises(ValueError):
        square_roi.get_std_error_of_the_mean(image=test_image)


def test_add_roi_to_image_unallowed_color():
    center = dict( x=6, y=6, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    square_roi = SquareRoi(center=center, height=3, width=3, pixel_size=voxel_data)
    with pytest.raises(ValueError):
        square_roi.add_roi_to_image(image=test_image, roi_color="Not a real colour")


def test_add_roi_to_image():
    roi_color = 'SkyBlue'
    expected = color.gray2rgb(test_image)
    expected[4:8, 4:8] = VALID_COLOURS[roi_color]

    center = dict(x=6, y=6, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    square_roi = SquareRoi(center=center, height=3, width=3, pixel_size=voxel_data)

    image_with_roi = square_roi.add_roi_to_image(image=test_image, roi_color=roi_color)

    assert image_with_roi.all() == expected.all()


def test_get_roi_part_of_image():
    expected = test_image[4:8, 4:8]

    center = dict(x=6, y=6, z=None)
    voxel_data = VoxelData(x=1.0, y=1.0, z=None)
    square_roi = SquareRoi(center=center, height=3, width=3, pixel_size=voxel_data)

    roi_part_of_image = square_roi.get_roi_part_of_image(image=test_image)

    assert roi_part_of_image.all() == expected.all()