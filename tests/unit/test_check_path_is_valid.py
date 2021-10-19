from pathlib import Path

import pytest

from dicom_image_tools.helpers.check_path_is_valid import check_path_is_valid_path


def test_check_path_is_valid_path_raises_TypeError_if_given_path_neither_Path_nor_str():
    with pytest.raises(TypeError):
        _ = check_path_is_valid_path(1234)


def test_check_path_is_valid_path_raises_TypeError_if_path_does_not_exist():
    with pytest.raises(TypeError):
        _ = check_path_is_valid_path(path_to_check=Path(__file__).parent / "SomeInvalidFilePath")


def test_check_path_is_valid_converts_valid_path_given_as_str_to_a_Path_instance():
    # Arrange
    valid_file_path = str(Path(__file__).absolute())

    # Act
    actual = check_path_is_valid_path(path_to_check=valid_file_path)

    # Assert
    assert isinstance(actual, Path)


def test_check_path_is_valid_returns_object_if_valid_Path_object():
    # Arrange
    expected = Path(__file__)

    # Act
    actual = check_path_is_valid_path(path_to_check=expected)

    # Assert
    assert actual == expected
