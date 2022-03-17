import tempfile

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pytest import fixture


@fixture(scope="function")
def image_with_negative_pixel_intensity_relationship():
    ds: Dataset = Dataset()
    ds.PixelIntensityRelationshipSign = -1
    ds.BitsStored = 12

    return {
        "image": np.array([[0, 0, 0], [2 ** 10, 2 ** 10, 2 ** 10]]),
        "normalized_image": np.multiply(
            np.array(
                [
                    [-(2 ** ds.BitsStored), -(2 ** ds.BitsStored), -(2 ** ds.BitsStored)],
                    [2 ** 10 - 2 ** ds.BitsStored, 2 ** 10 - 2 ** ds.BitsStored, 2 ** 10 - 2 ** ds.BitsStored],
                ]
            ),
            -1,
        ),
        "metadata": ds,
    }


@fixture(scope="function")
def dose_matrix() -> np.ndarray:
    return np.array(
        [
            [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
            [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
            [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        ]
    )


@fixture(scope="function")
def dose_matrix_tempfile(dose_matrix):
    suffix = ".dcm"
    filename_little_endian = tempfile.NamedTemporaryFile(suffix=suffix).name

    print("Setting file meta information...")
    # Populate required values for file meta information
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.2"  # RT Dose Storage
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"
    file_meta.TransferSyntaxUID = "1.2.840.10008.1.2"  # Implicit little endian

    print("Setting dataset values...")
    # Create the FileDataset instance (initially no data elements, but file_meta
    # supplied)
    ds = FileDataset(filename_little_endian, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.DoseGridScaling = 1
    ds.Modality = "RTDOSE"
    ds.Rows = dose_matrix.shape[0]
    ds.Columns = dose_matrix.shape[1]
    ds.BitsAllocated = 16
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = dose_matrix.tobytes()
    ds.save_as(filename_little_endian)

    return filename_little_endian


@fixture(scope="function")
def dose_matrix_file(dose_matrix_tempfile):
    return pydicom.dcmread(dose_matrix_tempfile)
