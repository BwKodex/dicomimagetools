from pathlib import Path

import numpy as np
import pytest

from dicom_image_tools.dicom_handlers.ct import CtSeries
from dicom_image_tools.dicom_handlers.dicom_import import import_dicom_from_folder
from dicom_image_tools.dicom_handlers.dicom_study import DicomStudy

CT_PATH1 = Path(__file__).parent.parent.parent / "test_data" / "ct_study" / "GE" / "serie1"


def _get_ct_study() -> DicomStudy:
    ct_study = import_dicom_from_folder(folder=CT_PATH1, recursively=False)
    return ct_study.get(list(ct_study.keys())[0])


def _get_ct_series() -> CtSeries:
    ct_study = _get_ct_study()
    ct_series = ct_study.Series[0]
    ct_series.import_image_volume()
    return ct_series


@pytest.fixture(scope="module")
def ct_series() -> CtSeries:
    return _get_ct_series()


@pytest.fixture(scope="module")
def ct_image_volume() -> np.ndarray:
    ct_series = _get_ct_series()
    return ct_series.ImageVolume


@pytest.fixture(scope="module")
def patient_mask_no_table():
    ct_series = _get_ct_series()
    ct_series.get_patient_mask(threshold=-500, remove_table=True)
    return ct_series.Mask
