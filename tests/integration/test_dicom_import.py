import logging
from pathlib import Path

import pytest
from pydicom.errors import InvalidDicomError

from dicom_image_tools.dicom_handlers.dicom_import import (
    import_dicom_file,
    import_dicom_from_folder,
)
from dicom_image_tools.dicom_handlers.dicom_study import DicomStudy
from dicom_image_tools.dicom_handlers.dose_matrix import DoseMatrix
from dicom_image_tools.helpers.voxel_data import VoxelData


def test_import_dicom_from_file_should_parse_kv_ma_ms_tags():

    file = Path(__file__).parent.parent / "test_data" / "io" / "iotest_60kv_7ma_20ms.dcm"

    expected_kv = 60.0
    expected_ma = 7.0
    expected_ms = 20.0

    dicom_study = import_dicom_file(file=file)
    dicom_study.Series[0].import_image()

    assert dicom_study.Series[0].kV[0] == expected_kv
    assert dicom_study.Series[0].mA[0] == expected_ma
    assert dicom_study.Series[0].ms[0] == expected_ms


def test_import_dicom_from_folder_should_find_all_files_in_given_folder():
    expected = 12

    folder = Path(__file__).parent.parent / "test_data"

    imported_dicom = import_dicom_from_folder(folder=folder, recursively=True)
    assert len(imported_dicom) == expected


def test_import_dicom_from_folder_should_ignore_DS_Store_files(caplog):
    folder = Path(__file__).parent.parent / "test_data" / "px"
    ds_store_file = folder / ".DS_Store"

    test_created_ds_store: bool = False
    try:
        if not ds_store_file.exists():
            with ds_store_file.open("w") as fp:
                fp.write("irrelevant string")

            test_created_ds_store = True

        with caplog.at_level(logging.DEBUG):
            _ = import_dicom_from_folder(folder=folder, recursively=True)
    finally:
        if test_created_ds_store:
            ds_store_file.unlink()

    assert "Skipping .DS_Store file" in caplog.text


def test_import_dicom_from_folder_imports_dose_reports_and_series():
    folder = Path(__file__).parent.parent / "test_data" / "ct_study" / "SeriesWithDoseReport"

    imported_dicom = import_dicom_from_folder(folder=folder, recursively=True)
    dcm_study = imported_dicom[list(imported_dicom.keys())[0]]

    assert len(dcm_study.DoseReports.Rdsr) == 1
    assert len(dcm_study.DoseReports.SecondaryCapture) == 3
    assert len(dcm_study.Series) == 1
    assert len(dcm_study.Series[0].FilePaths) == 60


def test_import_dicom_from_folder_raises_type_error():
    with pytest.raises(TypeError) as excinfo:
        # noinspection PyTypeChecker
        import_dicom_from_folder(folder=123)

    assert "Invalid path" in str(excinfo.value)


def test_import_dicom_from_folder_raises_value_error_if_not_directory():
    with pytest.raises(ValueError) as excinfo:
        import_dicom_from_folder(Path(__file__))

    assert "The given folder is not a directory" in str(excinfo.value)


def test_import_dicom_from_folder_raises_value_error_if_no_valid_dicom_files():
    with pytest.raises(ValueError) as excinfo:
        import_dicom_from_folder(folder=Path(__file__).parent)

    assert "The given folder contains no valid DICOM files" in str(excinfo.value)


def test_import_dicom_file():
    file = Path(__file__).parent.parent / "test_data" / "ct_study" / "GE" / "serie1" / "1"

    dicom_study = import_dicom_file(file=file)

    assert isinstance(dicom_study, DicomStudy)
    assert len(dicom_study.Series) == 1
    assert len(dicom_study.Series[0].CompleteMetadata) == 1


def test_import_dicom_file_should_accept_file_without_manufacturer_model_name():
    file = Path(__file__).parent.parent / "test_data" / "io" / "iotest_no_manufacturer_model.dcm"

    dicom_study = import_dicom_file(file=file)

    assert isinstance(dicom_study, DicomStudy)
    assert len(dicom_study.Series) == 1
    assert len(dicom_study.Series[0].CompleteMetadata) == 1


def test_import_dicom_file_should_set_voxel_size_to_1by1pixels_when_no_pixel_size_info():
    file = Path(__file__).parent.parent / "test_data" / "io" / "iotest_no_detector_spacing.dcm"

    dicom_study = import_dicom_file(file=file)

    assert isinstance(dicom_study, DicomStudy)
    assert len(dicom_study.Series) == 1
    assert len(dicom_study.Series[0].CompleteMetadata) == 1
    assert dicom_study.Series[0].VoxelData[0] == VoxelData(x=1, y=1, unit="pixels")


def test_import_dicom_file_should_accept_ct_files_without_manufacturer_model_name():
    filedir = Path(__file__).parent.parent / "test_data" / "ct_study" / "CtSeriesNoManufacturerModelName"

    dicom_studies = import_dicom_from_folder(folder=filedir)
    keys = list(dicom_studies.keys())

    assert len(keys) == 1

    dicom_study = dicom_studies[list(dicom_studies.keys())[0]]

    assert isinstance(dicom_study, DicomStudy)
    assert len(dicom_study.Series) == 1
    assert len(dicom_study.Series[0].FilePaths) == 4


def test_import_image_volume_should_replace_erase_series_data_before_appending_new_values():
    # Arrange
    filedir = Path(__file__).parent.parent / "test_data" / "ct_study" / "CtSeriesNoManufacturerModelName"

    dicom_studies = import_dicom_from_folder(folder=filedir)
    dicom_series = dicom_studies[list(dicom_studies.keys())[0]].Series[0]

    images = len(dicom_series.FilePaths)

    dicom_series.import_image_volume()

    expected = [
        len(dicom_series.kV),
        len(dicom_series.mA),
        len(dicom_series.CompleteMetadata),
        len(dicom_series.VoxelData),
        len(dicom_series.FilePaths),
        dicom_series.ImageVolume.shape[2]
    ]

    # Act
    dicom_series.import_image_volume()

    actual = [
        len(dicom_series.kV),
        len(dicom_series.mA),
        len(dicom_series.CompleteMetadata),
        len(dicom_series.VoxelData),
        len(dicom_series.FilePaths),
        dicom_series.ImageVolume.shape[2]
    ]

    # Assert
    assert actual == expected


def test_import_dicom_file_raises_type_error():
    with pytest.raises(TypeError) as excinfo:
        # noinspection PyTypeChecker
        import_dicom_file(file=1234)

    assert "Invalid path" in str(excinfo.value)


def test_import_dicom_file_raises_value_error_if_not_file():
    file = Path(__file__).parent.parent / "test_data" / "ct_study" / "GE" / "serie1"

    with pytest.raises(ValueError) as excinfo:
        _ = import_dicom_file(file=file)

    assert "File is not a valid file" in str(excinfo.value)


def test_import_dicom_file_raises_invalid_dicom_error():
    file = Path(__file__)

    with pytest.raises(InvalidDicomError):
        import_dicom_file(file=file)


def test_import_file_should_create_dose_matrix_series():
    # Arrange
    file = Path(__file__).parent.parent / "test_data" / "dose_matrix" / "dose_matrix_1.dcm"

    # Act
    actual = import_dicom_file(file=file)

    # Assert
    assert isinstance(actual.Series[0], DoseMatrix)


def test_import_dicom_from_folder_correctly_imports_dose_matrix_files():
    # Arrange
    folder = Path(__file__).parent.parent / "test_data" / "dose_matrix"

    # Act
    dicom_study = import_dicom_from_folder(folder=folder)
    actual = dicom_study[list(dicom_study.keys())[0]].Series[0]

    # Assert
    assert isinstance(actual, DoseMatrix)
