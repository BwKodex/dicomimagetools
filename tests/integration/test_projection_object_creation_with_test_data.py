from pathlib import Path

import pydicom
import pytest

from dicom_image_tools.dicom_handlers.dicom_import import import_dicom_from_folder
from dicom_image_tools.dicom_handlers.projection import ProjectionSeries


@pytest.fixture(scope="session")
def io_image():
    fp = Path(__file__).parent.parent / "test_data" / "io" / "iotest.dcm"
    dcm = pydicom.dcmread(fp=str(fp.absolute()))
    return fp, dcm


def test_projection_series_creation(io_image):
    fp, dcm = io_image
    proj_ser = ProjectionSeries(file=fp, dcm=dcm)

    assert proj_ser.Manufacturer == "Sirona Dental, Inc."
    assert proj_ser.ManufacturersModelName == "MAXIMUS SENSOR"


def test_projection_series_fail_add_other_series(io_image):
    fp, dcm = io_image
    fp_other = fp.parent.parent / "px" / "pxtest.dcm"
    proj_ser = ProjectionSeries(file=fp, dcm=dcm)

    with pytest.raises(ValueError):
        proj_ser.add_file(file=fp_other, dcm=None)


def test_projection_series_import_image(io_image):
    fp, dcm = io_image
    proj_ser = ProjectionSeries(file=fp, dcm=None)
    proj_ser.import_image()

    assert len(proj_ser.ImageVolume) == 1


def test_projection_series_sort_image_on_acquisition_time_should_reorder_file_paths():
    # Arrange
    folder = Path(__file__).parent.parent / "test_data" / "io" / "Edge-fantom"
    files = [
        folder / "Image9.dcm",
        folder / "Image5.dcm",
        folder / "Image10.dcm",
    ]

    proj_ser = ProjectionSeries(file=files[0], dcm=None)
    proj_ser.add_file(file=files[1], dcm=None)
    proj_ser.add_file(file=files[2], dcm=None)
    proj_ser.import_image()

    expected_file_paths = [files[2], files[0], files[1]]
    expected_complete_metadata = [
        proj_ser.CompleteMetadata[2],
        proj_ser.CompleteMetadata[0],
        proj_ser.CompleteMetadata[1],
    ]

    expected_image_volume = [proj_ser.ImageVolume[2], proj_ser.ImageVolume[0], proj_ser.ImageVolume[1]]

    # Act
    proj_ser.sort_images_on_acquisition_time()

    # Assert
    assert expected_file_paths == proj_ser.FilePaths
    assert expected_complete_metadata == proj_ser.CompleteMetadata
    assert expected_image_volume == proj_ser.ImageVolume
