from pathlib import Path
from pydicom.errors import InvalidDicomError
import pytest

from dicom_image_tools.dicom_handlers.dicom_import import import_dicom_from_folder, import_dicom_file
from dicom_image_tools.dicom_handlers.dicom_study import DicomStudy


def test_import_dicom_from_folder():
    folder = Path(__file__).parent.parent / 'test_data'

    imported_dicom = import_dicom_from_folder(folder=folder, recursively=True)
    assert len(imported_dicom) == 3


def test_import_dicom_from_folder_raises_type_error():
    with pytest.raises(TypeError) as excinfo:
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
    file = Path(__file__).parent.parent /'test_data' / 'ct_study' / 'GE' / 'serie1' / '1'

    dicom_study = import_dicom_file(file=file)

    assert isinstance(dicom_study, DicomStudy)
    assert len(dicom_study.Series) == 1


def test_import_dicom_file_raises_type_error():
    with pytest.raises(TypeError) as excinfo:
        import_dicom_file(file='InvalidFileType')

    assert 'file must be a Path object' in str(excinfo.value)


def test_import_dicom_file_raises_value_error_if_not_file():
    file = Path(__file__).parent.parent / 'test_data' / 'ct_study' / 'GE' / 'serie1'

    with pytest.raises(ValueError) as excinfo:
        dicom_study = import_dicom_file(file=file)

    assert 'File is not a valid file' in str(excinfo.value)


def test_import_dicom_file_raises_invalid_dicom_error():
    file = Path(__file__)

    with pytest.raises(InvalidDicomError):
        import_dicom_file(file=file)