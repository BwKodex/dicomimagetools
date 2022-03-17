from pathlib import Path

import numpy as np
import pytest

from dicom_image_tools.dicom_handlers.dicom_series import DicomSeries


def test_normalize_pixel_intensity_relationship_should_reverse_pixel_intensities_when_pixel_intensity_relationship_sign_is_negative(
    image_with_negative_pixel_intensity_relationship,
):
    # Arrange
    series = DicomSeries("TestSeries")
    series.CompleteMetadata.append(image_with_negative_pixel_intensity_relationship.get("metadata"))
    series.ImageVolume = [image_with_negative_pixel_intensity_relationship.get("image")]
    expected = image_with_negative_pixel_intensity_relationship.get("normalized_image")

    # Act
    series.normalize_pixel_intensity_relationship()

    # Assert
    assert np.equal(expected, series.ImageVolume).all()


def test_normalize_pixel_intensity_relationship_should_set_PixelIntensityNormailzed_to_True(
    image_with_negative_pixel_intensity_relationship,
):
    # Arrange
    series = DicomSeries("TestSeries")
    series.CompleteMetadata.append(image_with_negative_pixel_intensity_relationship.get("metadata"))
    series.ImageVolume = [image_with_negative_pixel_intensity_relationship.get("image")]

    assert series.PixelIntensityNormalized == False

    # Act
    series.normalize_pixel_intensity_relationship()

    # Assert
    assert series.PixelIntensityNormalized


def test_normalize_pixel_intensity_relationship_should_return_do_nothing_if_PixelIntensityNormailzed_is_True(
    image_with_negative_pixel_intensity_relationship,
):
    # Arrange
    series = DicomSeries("TestSeries")
    series.CompleteMetadata.append(image_with_negative_pixel_intensity_relationship.get("metadata"))
    series.ImageVolume = [image_with_negative_pixel_intensity_relationship.get("image")]
    series.PixelIntensityNormalized = True
    expected = image_with_negative_pixel_intensity_relationship.get("image").copy()

    # Act
    series.normalize_pixel_intensity_relationship()

    # Assert
    assert np.equal(expected, series.ImageVolume).all()


def test_normalize_pixel_intensity_relationship_should_return_do_nothing_if_PixelIntensityRelationshipSign_is_1(
    image_with_negative_pixel_intensity_relationship,
):
    # Arrange
    series = DicomSeries("TestSeries")
    series.CompleteMetadata.append(image_with_negative_pixel_intensity_relationship.get("metadata"))
    series.ImageVolume = [image_with_negative_pixel_intensity_relationship.get("image")]
    series.CompleteMetadata[0].PixelIntensityRelationshipSign = 1
    expected = image_with_negative_pixel_intensity_relationship.get("image").copy()

    # Act
    series.normalize_pixel_intensity_relationship()

    # Assert
    assert np.equal(expected, series.ImageVolume).all()


def test_normalize_pixel_intensity_relationship_raises_ValueError_if_image_volume_is_not_imported(
    image_with_negative_pixel_intensity_relationship,
):
    # Arrange
    series = DicomSeries("TestSeries")
    series.CompleteMetadata.append(image_with_negative_pixel_intensity_relationship.get("metadata"))

    with pytest.raises(ValueError):
        series.normalize_pixel_intensity_relationship()


def test_dicom_series_show_image_throws_TypeError_if_given_index_is_not_an_integer(
    image_with_negative_pixel_intensity_relationship,
):
    # Arrange
    series = DicomSeries("TestSeries")
    series.FilePaths.append(Path(__file__))

    # Assert
    with pytest.raises(TypeError) as exc:
        series.show_image(index="InvalidIndexType")

    assert "Image index must be given as an integer" in str(exc.value)


def test_dicom_series_show_image_throws_TypeError_if_given_rois_is_not_a_list_of_rois(
    image_with_negative_pixel_intensity_relationship,
):
    # Arrange
    series = DicomSeries("TestSeries")
    series.FilePaths.append(Path(__file__))

    # Assert
    with pytest.raises(TypeError) as exc:
        series.show_image(rois=[1, 2, 3, 4])

    assert "Only list of SquareRoi instances are implemented for plotting" in str(exc.value)


def test_dicom_series_show_image_throws_ValueError_if_given_index_out_of_valid_range(
    image_with_negative_pixel_intensity_relationship,
):
    # Arrange
    series = DicomSeries("TestSeries")
    series.FilePaths.append(Path(__file__))

    # Assert
    with pytest.raises(ValueError) as exc:
        series.show_image(index=-2)

    assert "Invalid image index specified" in str(exc.value)


def test_dicom_series_show_image_throws_TypeError_if_given_colour_map_is_not_a_str(
    image_with_negative_pixel_intensity_relationship,
):
    # Arrange
    series = DicomSeries("TestSeries")
    series.FilePaths.append(Path(__file__))

    # Assert
    with pytest.raises(TypeError) as exc:
        series.show_image(colour_map=123)

    assert "The colour map must be given as the name of an implemented colour map. Allowed values are" in str(exc.value)


def test_dicom_series_show_image_throws_NotImplementedError_for_invalid_colour_map(
    image_with_negative_pixel_intensity_relationship,
):
    # Arrange
    series = DicomSeries("TestSeries")
    series.FilePaths.append(Path(__file__))

    # Assert
    with pytest.raises(NotImplementedError) as exc:
        series.show_image(colour_map="InvalidColourMap")

    assert "Colour map must be one of the following implemented colour maps: " in str(exc.value)


def test_dicom_series_show_image_throws_TypeError_for_invalid_window_type(
    image_with_negative_pixel_intensity_relationship,
):
    # Arrange
    series = DicomSeries("TestSeries")
    series.FilePaths.append(Path(__file__))

    # Assert
    with pytest.raises(TypeError) as exc:
        series.show_image(window=12)

    assert "If specified, the window must be a tuple of length 2" in str(exc.value)


def test_dicom_series_show_image_throws_ValueError_for_invalid_window_length(
    image_with_negative_pixel_intensity_relationship,
):
    # Arrange
    series = DicomSeries("TestSeries")
    series.FilePaths.append(Path(__file__))

    # Assert
    with pytest.raises(ValueError) as exc:
        series.show_image(window=(12, 32, 12))

    assert "The specified window must be a tuple of exactly 2 floats" in str(exc.value)
