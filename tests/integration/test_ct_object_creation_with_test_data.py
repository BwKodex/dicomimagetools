from pathlib import Path

import pytest
from numpy import floor

from dicom_image_tools.dicom_handlers.ct import CtSeries


@pytest.fixture()
def example_data_path_fixture():
    base_dir = Path(__file__).parent.parent / "test_data"
    example_data_paths = {
        "ct": base_dir / "ct_study",
    }
    return example_data_paths


def test_ct_series_add_file(example_data_path_fixture):
    ct_series = CtSeries(series_instance_uid="1.2.826.0.1.3680043.8.971.31305363770056566540494760179678687617")
    ct_series.add_file(file=example_data_path_fixture["ct"] / "GE" / "serie1" / "1")

    assert len(ct_series.FilePaths) == 1


def test_ct_series_add_file_sorts_on_image_position_patient_when_slice_location_missing(example_data_path_fixture):
    # Arrange
    expected = "1"
    ct_series = CtSeries(series_instance_uid="1.2.826.0.1.3680043.8.971.31639893030722307093467532646480365648")

    fp_first_slice = example_data_path_fixture["ct"] / "GE" / "MissingSliceLocationSeries" / "1"
    fp_last_slice = example_data_path_fixture["ct"] / "GE" / "MissingSliceLocationSeries" / "2"

    # Act
    ct_series.add_file(file=fp_last_slice)
    ct_series.add_file(file=fp_first_slice)

    actual = ct_series.FilePaths[0].name

    # Assert
    assert actual == expected


def test_ct_series_raises_error_on_adding_file_from_other_series(example_data_path_fixture):
    ct_series = CtSeries(series_instance_uid="WrongSeriesInstanceUid")
    with pytest.raises(ValueError):
        ct_series.add_file(file=example_data_path_fixture["ct"] / "GE" / "serie1" / "1")


def test_ct_series_import_image_volume(example_data_path_fixture):
    ct_series = CtSeries(series_instance_uid="1.2.826.0.1.3680043.8.971.31305363770056566540494760179678687617")
    ct_series.add_file(file=example_data_path_fixture["ct"] / "GE" / "serie1" / "1")
    ct_series.import_image_volume()

    assert len(ct_series.CompleteMetadata) == 1
    assert ct_series.Manufacturer == "GE MEDICAL SYSTEMS"
    assert ct_series.kV == [120]
    assert ct_series.mA == [520]
    assert ct_series.SlicePosition == [-7.5]
    assert ct_series.ImageVolume.shape == (512, 512, 1)


def test_ct_series_pixel_data_removed_from_complete_metadata(example_data_path_fixture):
    ct_series = CtSeries(series_instance_uid="1.2.826.0.1.3680043.8.971.31305363770056566540494760179678687617")
    ct_series.add_file(file=example_data_path_fixture["ct"] / "GE" / "serie1" / "1")
    ct_series.import_image_volume()

    with pytest.raises(KeyError):
        tmp = ct_series.CompleteMetadata[0][0x7FE00010]


def test_ct_series_get_patient_mask_wrong_threshold_type(example_data_path_fixture):
    ct_series = CtSeries(series_instance_uid="1.2.826.0.1.3680043.8.971.31305363770056566540494760179678687617")
    ct_series.add_file(file=example_data_path_fixture["ct"] / "GE" / "serie1" / "1")
    ct_series.import_image_volume()

    with pytest.raises(TypeError):
        ct_series.get_patient_mask(threshold="WrongThresholdType", remove_table=False)


def test_ct_series_get_patient_mask(example_data_path_fixture):
    ct_series = CtSeries(series_instance_uid="1.2.826.0.1.3680043.8.971.31305363770056566540494760179678687617")
    ct_series.add_file(file=example_data_path_fixture["ct"] / "GE" / "serie1" / "1")
    ct_series.import_image_volume()
    ct_series.get_patient_mask(threshold=-500, remove_table=False)

    assert ct_series.MaskSuccess is True


def test_ct_series_get_patient_mask_remove_table(example_data_path_fixture):
    ct_series = CtSeries(series_instance_uid="1.2.826.0.1.3680043.8.971.31305363770056566540494760179678687617")
    ct_series.add_file(file=example_data_path_fixture["ct"] / "GE" / "serie1" / "1")
    ct_series.add_file(file=example_data_path_fixture["ct"] / "GE" / "serie1" / "2")
    ct_series.import_image_volume()
    ct_series.get_patient_mask(threshold=-500, remove_table=True)

    assert ct_series.MaskSuccess is True
    assert ct_series.Mask is not None
    assert sum(ct_series.Mask.flat) > 0
    assert ct_series.MaskSuccess is True
    assert ct_series.PatientClipped is False
    assert ct_series.MedianHuPatientVolume == 125.0
    assert floor(ct_series.MeanHuPatientVolume) == 85.0
