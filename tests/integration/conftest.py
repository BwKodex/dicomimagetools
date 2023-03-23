import shutil
from pathlib import Path

import pytest

from dicom_image_tools.dicom_handlers.dicom_import import import_dicom_file
from dicom_image_tools.dicom_handlers.projection import ProjectionSeries

IO_FILE_PATH1_SIRONA = Path(__file__).parent.parent / "test_data" / "io" / "iotest.dcm"


@pytest.fixture(scope="function")
def io_series_sirona() -> ProjectionSeries:
    io_study = import_dicom_file(IO_FILE_PATH1_SIRONA)
    return io_study.Series[0]


@pytest.fixture(scope="session")
def file_creation_dir() -> Path:
    temp_dir = Path(__file__).parent / "Temptestfiles"
    temp_dir.mkdir(exist_ok=False, parents=False)

    yield temp_dir

    shutil.rmtree(temp_dir)
