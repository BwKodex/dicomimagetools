from dicom_image_tools.dicom_handlers.dicom_study import DicomStudy, DicomSeries
from pathlib import Path
import pytest


@pytest.fixture()
def example_data_path_fixture():
    base_dir = Path(__file__).parent.parent / 'test_data'
    example_data_paths = {
        "ct": base_dir / 'ct_study',
    }
    return example_data_paths


def test_dicom_series_add_file(example_data_path_fixture):
    dcm_series = DicomSeries(series_instance_uid='1.2.826.0.1.3680043.8.971.31305363770056566540494760179678687617')
    dcm_series.add_file(file=example_data_path_fixture['ct'] / 'serie1' / '1')

    assert len(dcm_series.FilePaths) == 1


def test_dicom_series_add_file_already_in_series_ignores_file(example_data_path_fixture):
    dcm_series = DicomSeries(series_instance_uid='1.2.826.0.1.3680043.8.971.31305363770056566540494760179678687617')
    dcm_file_path = example_data_path_fixture['ct'] / 'serie1' / '1'

    dcm_series.add_file(file=dcm_file_path)
    dcm_series.add_file(file=dcm_file_path)

    assert len(dcm_series.FilePaths) == 1


def test_dicom_series_add_file_wrong_series_instance_uid(example_data_path_fixture):
    dcm_series = DicomSeries(series_instance_uid='some-uid')

    with pytest.raises(ValueError):
        dcm_series.add_file(file=example_data_path_fixture['ct'] / 'serie1' / '1')


def test_dicom_study_add_file(example_data_path_fixture):
    expected_series_instance_uid = '1.2.826.0.1.3680043.8.971.31305363770056566540494760179678687617'

    dcm_study = DicomStudy('1.2.826.0.1.3680043.8.971.19037936343369135096938015896747654128')
    dcm_study.add_file(file=example_data_path_fixture['ct'] / 'serie1' / '1')

    assert len(dcm_study.Series) == 1
    assert dcm_study.Series[0].SeriesInstanceUid == expected_series_instance_uid


def test_dicom_study_add_file_wrong_study_instance_uid(example_data_path_fixture):
    dcm_study = DicomStudy('some-uid')
    with pytest.raises(ValueError):
        dcm_study.add_file(file=example_data_path_fixture['ct'] / 'serie1' / '1')

