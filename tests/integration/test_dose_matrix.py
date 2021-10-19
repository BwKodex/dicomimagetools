from pathlib import Path

import pydicom
import pytest

from dicom_image_tools.dicom_handlers.dose_matrix import DoseMatrix
from dicom_image_tools.helpers.geometry import get_shortest_line_point_dist, Line
from dicom_image_tools.helpers.point import Point


def test_DoseMatrix_does_not_add_file_if_duplicate_of_already_added_file():
    # Arrange
    filepath = Path(__file__).parent.parent / "test_data" / "dose_matrix" / "dose_matrix_1.dcm"
    expected = 1

    # Act
    dose_matrix = DoseMatrix(file=filepath)
    dose_matrix.add_file(file=filepath)

    actual = len(dose_matrix.CompleteMetadata)

    # Assert
    assert actual == expected


def test_DoseMatrix_deletes_pixel_data_from_metadata_in_complete_metadata_property():
    # Arrange
    filepath = Path(__file__).parent.parent / "test_data" / "dose_matrix" / "dose_matrix_1.dcm"
    dcm = pydicom.dcmread(filepath)

    # Act
    dose_matrix = DoseMatrix(file=filepath, dcm=dcm)

    # Assert
    assert "PixelData" not in dose_matrix.CompleteMetadata[0]


def test_get_shortes_line_point_dist_raises_TypeError_when_line_is_not_a_line_object():
    # Arrange
    point = Point(x=0.1, y=0.1)
    # Act
    with pytest.raises(TypeError):
        _ = get_shortest_line_point_dist(line="InvalidLine", point=point)


def test_get_shortes_line_point_dist_raises_TypeError_when_point_is_not_a_valid_point_object():
    # Arrange
    line = Line(x0=0, y0=0, x1=1, y1=1)

    # Act
    with pytest.raises(TypeError):
        _ = get_shortest_line_point_dist(line=line, point="SomeInvalidPoint")
