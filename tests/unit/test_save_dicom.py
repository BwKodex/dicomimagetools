from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
import pydicom
import pytest

from dicom_image_tools.dicom_handlers.save_dicom import save_dicom


def test_save_dicom_throws_type_error_when_invalid_output_path_type():
    expected_message = "The output path must be a Path or a string object"

    with pytest.raises(TypeError) as exc_info:
        save_dicom(
            image=np.ndarray([1, 2, 3]),
            metadata=pydicom.dataset.FileDataset(BytesIO(), pydicom.dataset.Dataset()),
            output_path=123
        )

    assert expected_message in str(exc_info.value)


def test_save_dicom_throws_value_error_when_output_path_does_not_exist():
    expected_message = "The output directory does not exist"

    with pytest.raises(ValueError) as exc_info:
        save_dicom(
            image=np.ndarray([1, 2, 3]),
            metadata=pydicom.dataset.FileDataset(BytesIO(), pydicom.dataset.Dataset()),
            output_path=Path(r"W:\SomeNoneExistingPath\That\I\Just\Made\Up.BwKodexFileFormat")
        )

    assert expected_message in str(exc_info.value)

