from dicom_image_tools.helpers.point import Point
from dicom_image_tools.roi.roi import Roi
import pytest


def test_roi_creation_wrong_type_center():
    with pytest.raises(TypeError):
        Roi(center=False)


def test_roi_creation_wrong_format_z_center():
    with pytest.raises(ValueError):
        Roi(center={'x': 1, 'y': 1})


def test_roi_creation_wrong_format_y_center():
    with pytest.raises(ValueError):
        Roi(center={'x': 1, 'z': 1})


def test_roi_creation_wrong_format_x_center():
    with pytest.raises(ValueError):
        Roi(center={'z': 1, 'y': 1})


def test_roi_creation_wrong_format_x_none_dict_center():
    with pytest.raises(TypeError):
        Roi(center={'x': None, 'y': 1, 'z': 1})


def test_roi_creation_wrong_format_y_dict_none_center():
    with pytest.raises(TypeError):
        Roi(center={'x': 1, 'y': None, 'z': 1})


def test_roi_creation_z_none_center():
    roi = Roi(center={'x': 1, 'y': 1, 'z': None})

    assert roi.Center.x == 1.0
    assert roi.Center.y == 1.0
    assert roi.Center.z is None


def test_roi_creation_z_has_value():
    roi = Roi(center={'x': 1, 'y': 1, 'z': 1})

    assert roi.Center.x == 1.0
    assert roi.Center.y == 1.0
    assert roi.Center.z == 1.0


def test_roi_creation_point_z_none_value():
    point = Point(x=1.0, y=1.0, z=None)
    roi = Roi(center=point)

    assert roi.Center.x == 1.0
    assert roi.Center.y == 1.0
    assert roi.Center.z is None


def test_roi_creation_point_z_has_value():
    point = Point(x=1.0, y=1.0, z=1.0)
    roi = Roi(center=point)

    assert roi.Center.x == 1.0
    assert roi.Center.y == 1.0
    assert roi.Center.z == 1.0


def test_roi_creation_point_y_none_value():
    point = Point(x=1.0, y=None, z=None)
    with pytest.raises(ValueError):
        Roi(center=point)
