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
