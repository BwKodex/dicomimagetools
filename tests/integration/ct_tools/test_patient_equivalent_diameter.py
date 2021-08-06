from typing import Tuple

import numpy as np
import pytest

from dicom_image_tools.ct_tools.patient_equivalent_diameter import (
    EquivalentDiameterData,
    SinogramData,
    _calculate_slice_area_equivalent_diameter,
    calculate_area_equivalent_diameter,
)
from dicom_image_tools.dicom_handlers.ct import CtSeries
from dicom_image_tools.helpers.voxel_data import VoxelData


def _get_slice_area_data(ct_series: CtSeries, patient_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, VoxelData]:
    im = ct_series.ImageVolume[:, :, int(round(ct_series.ImageVolume.shape[2] / 2))].copy()
    mask = patient_mask[:, :, int(round(ct_series.ImageVolume.shape[2] / 2))].copy()
    voxel_data = ct_series.VoxelData[int(round(ct_series.ImageVolume.shape[2] / 2))]

    return im, mask, voxel_data


def test_calculate_slice_area_equivalent_diameter_raises_typeerror_image(ct_series, patient_mask_no_table):
    im, mask, voxel_data = _get_slice_area_data(ct_series=ct_series, patient_mask=patient_mask_no_table)
    with pytest.raises(TypeError):
        _calculate_slice_area_equivalent_diameter(image="Invalid type", mask=mask, voxel_data=voxel_data)


def test_calculate_slice_area_equivalent_diameter_raises_typeerror_mask(ct_series, patient_mask_no_table):
    im, mask, voxel_data = _get_slice_area_data(ct_series=ct_series, patient_mask=patient_mask_no_table)
    with pytest.raises(TypeError):
        _calculate_slice_area_equivalent_diameter(image=im, mask="Invalid type", voxel_data=voxel_data)


def test_calculate_slice_area_equivalent_diameter_raises_typeerror_voxel_data(ct_series, patient_mask_no_table):
    im, mask, voxel_data = _get_slice_area_data(ct_series=ct_series, patient_mask=patient_mask_no_table)
    with pytest.raises(TypeError):
        _calculate_slice_area_equivalent_diameter(image=im, mask=mask, voxel_data="Invalid type")


def test_calculate_slice_area_equivalent_diameter_returns_equivalent_diameter_data(ct_series, patient_mask_no_table):
    im, mask, voxel_data = _get_slice_area_data(ct_series=ct_series, patient_mask=patient_mask_no_table)
    actual = _calculate_slice_area_equivalent_diameter(image=im, mask=mask, voxel_data=voxel_data)

    assert isinstance(actual, EquivalentDiameterData)


def test_calculate_area_equivalent_diameter(ct_series):
    expected = EquivalentDiameterData(
        Area_px=152927.0,
        EAD_px=441.2626245873473,
        EAD_cm=21.546015559613455,
        EquivalentAreaCircumference_cm=67.688804196213,
        MeanHU=86.66725300306682,
        MedianHU=125.0,
        LAT_cm=21.5820202,
        AP_cm=21.5331921,
        WED_px=459.9868683189358,
        WED_cm=22.46028480496383,
        EED_px=441.4997168742014,
        EED_cm=21.557592325505194,
    )

    actual, _, _ = calculate_area_equivalent_diameter(ct_series)

    assert len(actual) == 4
    assert actual[0] == expected


def test_calculate_area_equivalent_diameter_and_radon(ct_series):
    expected = SinogramData(
        MaxPixels=444,
        MinPixels=441,
        MaxCm=21.679676399999998,
        MinCm=21.5331921,
        MaxAngle=1.0,
        MinAngle=90.0,
        EquivalentDiameter=21.60631011040609,
    )

    _, actual, _ = calculate_area_equivalent_diameter(ct=ct_series, use_radon=True, use_hough=False)

    assert len(actual) == 4
    assert actual[0] == expected


def test_calculate_area_equivalent_diameter_and_hough(ct_series):
    expected = SinogramData(
        MaxPixels=444,
        MinPixels=441,
        MaxCm=21.679676399999998,
        MinCm=21.5331921,
        MaxAngle=1.0,
        MinAngle=90.0,
        EquivalentDiameter=21.60631011040609,
    )

    _, _, actual = calculate_area_equivalent_diameter(ct=ct_series, use_radon=False, use_hough=True)

    assert len(actual) == 4
    assert actual[0] == expected
