from uuid import uuid4

from dicom_image_tools.dicom_handlers.save_dicom import save_dicom


def test_save_dicom_stores_image_when_correct_data_is_given(io_series_sirona, file_creation_dir):
    # Arrange
    output_filepath = file_creation_dir / f"{uuid4()}.dcm"
    io_series_sirona.import_image()

    # Act
    save_dicom(
        image=io_series_sirona.ImageVolume[0],
        metadata=io_series_sirona.CompleteMetadata[0],
        output_path=output_filepath
    )

    # Assert
    assert output_filepath.exists()
    output_filepath.unlink()
