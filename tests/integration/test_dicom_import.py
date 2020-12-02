import logging
from pathlib import Path
from pydicom.errors import InvalidDicomError
import pytest

from dicom_image_tools.dicom_handlers.dicom_import import import_dicom_from_folder, import_dicom_file
from dicom_image_tools.dicom_handlers.dicom_study import DicomStudy

def test_import_dicom_from_file_should_parse_kv_ma_ms_tags():

    file = Path(__file__).parent.parent / 'test_data' / 'io' / 'iotest_60kv_7ma_20ms.dcm'
    
    expected_kv = 60.0
    expected_ma = 7.0
    expected_ms = 20.0
        
    dicom_study = import_dicom_file(file=file)
    dicom_study.Series[0].import_image()

    assert dicom_study.Series[0].kV[0] == expected_kv
    assert dicom_study.Series[0].mA[0] == expected_ma
    assert dicom_study.Series[0].ms[0] == expected_ms


def test_import_dicom_from_folder_should_find_all_files_in_given_folder():
    expected = 7

    folder = Path(__file__).parent.parent / 'test_data'

    imported_dicom = import_dicom_from_folder(folder=folder, recursively=True)
    assert len(imported_dicom) == expected


def test_import_dicom_from_folder_should_ignore_DS_Store_files(caplog):
    folder = Path(__file__).parent.parent / 'test_data' / 'px'
    ds_store_file = folder / '.DS_Store'

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
            ds_store_file.unlink(missing_ok=True)

    assert "Skipping .DS_Store file" in caplog.text


def test_import_dicom_from_folder_imports_dose_reports_and_series():
    folder = Path(__file__).parent.parent / 'test_data' / 'ct_study' / 'SeriesWithDoseReport'

    imported_dicom = import_dicom_from_folder(folder=folder, recursively=True)
    dcm_study = imported_dicom[list(imported_dicom.keys())[0]]

    assert len(dcm_study.DoseReports.Rdsr) == 1
    assert len(dcm_study.DoseReports.SecondaryCapture) == 3
    assert len(dcm_study.Series) == 1
    assert len(dcm_study.Series[0].FilePaths) == 60


def test_import_dicom_from_folder_raises_type_error():
    with pytest.raises(TypeError) as excinfo:
        # noinspection PyTypeChecker
        import_dicom_from_folder(folder='InvalidFolderType')

    assert 'folder must be a Path object' in str(excinfo.value)


def test_import_dicom_from_folder_raises_value_error_if_not_directory():
    with pytest.raises(ValueError) as excinfo:
        import_dicom_from_folder(Path(__file__))

    assert 'The given folder is not a directory' in str(excinfo.value)


def test_import_dicom_from_folder_raises_value_error_if_no_valid_dicom_files():
    with pytest.raises(ValueError) as excinfo:
        import_dicom_from_folder(folder=Path(__file__).parent)

    assert 'The given folder contains no valid DICOM files' in str(excinfo.value)


def test_import_dicom_file():
    file = Path(__file__).parent.parent / 'test_data' / 'ct_study' / 'GE' / 'serie1' / '1'

    dicom_study = import_dicom_file(file=file)

    assert isinstance(dicom_study, DicomStudy)
    assert len(dicom_study.Series) == 1
    assert len(dicom_study.Series[0].CompleteMetadata) == 1


def test_import_dicom_file_raises_type_error():
    with pytest.raises(TypeError) as excinfo:
        # noinspection PyTypeChecker
        import_dicom_file(file='InvalidFileType')

    assert 'file must be a Path object' in str(excinfo.value)


def test_import_dicom_file_raises_value_error_if_not_file():
    file = Path(__file__).parent.parent / 'test_data' / 'ct_study' / 'GE' / 'serie1'

    with pytest.raises(ValueError) as excinfo:
        _ = import_dicom_file(file=file)

    assert 'File is not a valid file' in str(excinfo.value)


def test_import_dicom_file_raises_invalid_dicom_error():
    file = Path(__file__)

    with pytest.raises(InvalidDicomError):
        import_dicom_file(file=file)
