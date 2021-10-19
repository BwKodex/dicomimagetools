import pytest

from dicom_image_tools.dicom_handlers.dose_matrix import DoseMatrix


def test_DoseMatrix_initialization_raises_value_error_on_when_no_series_instance_uid(dose_matrix_tempfile, dose_matrix_file):
    with pytest.raises(ValueError):
        _ = DoseMatrix(dose_matrix_tempfile, None)
