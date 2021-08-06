import pytest

from dicom_image_tools.dicom_handlers.dicom_study import DicomSeries, DicomStudy


def test_dicom_study_creation():
    expected = dict(type=DicomStudy, study_instance_uid="test-uid")
    dcm_obj = DicomStudy(study_instance_uid=expected["study_instance_uid"])

    assert isinstance(dcm_obj, expected.get("type"))
    assert dcm_obj.StudyInstanceUid == expected["study_instance_uid"]


def test_dicom_study_creation_wrong_study_instance_uid_type():
    with pytest.raises(TypeError):
        dcm_obj = DicomStudy(study_instance_uid=12345)


def test_dicom_series_creation():
    expected = dict(type=DicomSeries, series_instance_uid="test-uid")
    dcm_obj = DicomSeries(series_instance_uid=expected["series_instance_uid"])

    assert isinstance(dcm_obj, expected.get("type"))
    assert dcm_obj.SeriesInstanceUid == expected["series_instance_uid"]


def test_dicom_study_creation_wrong_series_instance_uid_type():
    with pytest.raises(TypeError):
        dcm_obj = DicomSeries(series_instance_uid=12345)
