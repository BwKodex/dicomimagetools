from pathlib import Path
import pydicom
import pytest

from dicom_image_tools.dicom_handlers.projection import ProjectionSeries


@pytest.fixture(scope='session')
def io_image():
    fp = Path(__file__).parent.parent / 'test_data' / 'io' / 'iotest.dcm'
    dcm = pydicom.dcmread(fp=str(fp.absolute()))
    return fp, dcm


def test_projection_series_creation(io_image):
    fp, dcm = io_image
    proj_ser = ProjectionSeries(file=fp, dcm=dcm)

    assert proj_ser.Manufacturer == 'Sirona Dental, Inc.'
    assert proj_ser.ManufacturersModelName == 'MAXIMUS SENSOR'


def test_projection_series_fail_add_other_series(io_image):
    fp, dcm = io_image
    fp_other = fp.parent.parent / 'px' / 'pxtest.dcm'
    proj_ser = ProjectionSeries(file=fp, dcm=dcm)

    with pytest.raises(ValueError):
        proj_ser.add_file(file=fp_other, dcm=None)


def test_projection_series_import_image(io_image):
    fp, dcm = io_image
    proj_ser = ProjectionSeries(file=fp, dcm=None)
    proj_ser.import_image()

    assert len(proj_ser.ImageVolume) == 1
