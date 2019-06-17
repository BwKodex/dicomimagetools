from dicom_image_tools.ct_tools.patient_equivalent_diameter import (
    calculate_area_equivalent_diameter, calculate_max_min_lat_ap_hough, calculate_max_min_lat_ap_radon, SinogramData)
from dicom_image_tools.helpers.voxel_data import VoxelData
from dicom_image_tools.dicom_handlers.ct import CtSeries
import numpy as np
import pytest


def test_calculate_area_equivalent_diameter_raises_typeerror_ct():
    with pytest.raises(TypeError):
        calculate_area_equivalent_diameter(ct='Invalid argument')


def test_calculate_area_equivalent_diameter_raises_typeerror_use_radon():
    with pytest.raises(TypeError):
        calculate_area_equivalent_diameter(ct=CtSeries(series_instance_uid="some_UID"), use_radon=1, use_hough=False)


def test_calculate_area_equivalent_diameter_raises_typeerror_use_hough():
    with pytest.raises(TypeError):
        calculate_area_equivalent_diameter(ct=CtSeries(series_instance_uid="some_UID"), use_radon=False, use_hough=1)


def test_calculate_max_min_lat_ap_hough_raises_typeerror_mask():
    voxel_data = VoxelData(x=1.0, y=1.0)
    with pytest.raises(TypeError):
        calculate_max_min_lat_ap_hough(mask="Invalid type", voxel_data=voxel_data)


def test_calculate_max_min_lat_ap_hough_raises_typeerror_mask_shape():
    invalid_matrix = np.array([
        [[0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0]]
    ])
    voxel_data = VoxelData(x=1.0, y=1.0)
    with pytest.raises(TypeError):
        calculate_max_min_lat_ap_hough(mask=invalid_matrix, voxel_data=voxel_data)


def test_calculate_max_min_lat_ap_hough():
    matrix = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
    ])

    expected = SinogramData(
        MaxAngle=90,
        MaxCm=0.9,
        MaxPixels=9,
        MinAngle=0,
        MinCm=0.5,
        MinPixels=5,
        EquivalentDiameter=np.sqrt(np.multiply(0.9, 0.5))
    )

    voxel_data = VoxelData(x=1.0, y=1.0)

    actual = calculate_max_min_lat_ap_hough(mask=matrix, voxel_data=voxel_data)

    assert actual == expected


def test_calculate_max_min_lat_ap_radon_raises_typeerror_mask():
    voxel_data = VoxelData(x=1.0, y=1.0)
    with pytest.raises(TypeError):
        calculate_max_min_lat_ap_radon(mask="Invalid type", voxel_data=voxel_data)


def test_calculate_max_min_lat_ap_radon_raises_typeerror_mask_shape():
    invalid_matrix = np.array([
        [[0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0]]
    ])
    voxel_data = VoxelData(x=1.0, y=1.0)
    with pytest.raises(TypeError):
        calculate_max_min_lat_ap_radon(mask=invalid_matrix, voxel_data=voxel_data)


def test_calculate_max_min_lat_ap_radon():
    matrix = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
    ])

    expected = SinogramData(
        MaxAngle=90,
        MaxCm=0.9,
        MaxPixels=9,
        MinAngle=0,
        MinCm=0.5,
        MinPixels=5,
        EquivalentDiameter=np.sqrt(np.multiply(0.9, 0.5))
    )

    voxel_data = VoxelData(x=1.0, y=1.0)

    actual = calculate_max_min_lat_ap_radon(mask=matrix, voxel_data=voxel_data)

    assert actual == expected
