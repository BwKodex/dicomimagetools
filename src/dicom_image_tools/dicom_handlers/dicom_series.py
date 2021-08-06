from pathlib import Path
from typing import List, Optional

import numpy as np
import pydicom
from pydicom import FileDataset

from ..helpers.voxel_data import VoxelData


class DicomSeries:
    """A class to manage DICOM files connected by a Series Instance UID

    Args:
        series_instance_uid: Series instance UID of the object to be created

    Attributes:
        SeriesInstanceUid: Series instance UID of the object
        FilePaths: Paths to the files added to the object
        CompleteMetadata: The complete set of metadata for the added files
        VoxelData: Voxel size information for included image files
        ImageVolume: The Image volume of the DICOM series
        Mask: A mask of the same dimension as the image volume to apply to the image volume

    """

    def __init__(self, series_instance_uid: str):
        if not isinstance(series_instance_uid, str):
            raise TypeError("series_instance_uid must be a string")
        self.FilePaths: List[Path] = []

        # Metadata
        self.SeriesInstanceUid: str = series_instance_uid
        self.SeriesDescription: Optional[str] = None
        self.CompleteMetadata: List[Optional[FileDataset]] = []
        self.VoxelData: List[VoxelData] = []
        self.PixelIntensityNormalized: bool = False

        self.ImageVolume: Optional[np.ndarray] = None
        self.Mask: Optional[np.ndarray] = None

    def add_file(self, file: Path, dcm: Optional[FileDataset] = None):
        """Add a file to the objects list of files

        First performs a check that the file is a valid DICOM file and that it belongs to the object/series

        Args:
            file: Path to where the file to be added is stored on disk
            dcm: The DICOM-file imported to a FileDataset object

        Raises:
            ValueError: if SeriesInstanceUID of the file is not the same as the SeriesInstanceUid attribute

        """
        if any([True if obj == file else False for obj in self.FilePaths]):
            # Return None since the file is already in hte volume
            return

        if dcm is None:
            dcm = pydicom.dcmread(fp=str(file.absolute()), stop_before_pixels=True)

        if dcm.SeriesInstanceUID != self.SeriesInstanceUid:
            msg = f"Wrong SeriesInstanceUID. Expected: {self.SeriesInstanceUid}; Input: {dcm.SeriesInstanceUID}"
            raise ValueError(msg)

        if "SeriesDescription" in dcm:
            self.SeriesDescription = dcm.SeriesDescription

        self.FilePaths.append(file)

    def normalize_pixel_intensity_relationship(self):
        """Reverse the pixel intensity for images with negative pixel intensity relationship to make the lower pixel
        value correspond to less X-Ray beam intensity

        Raises:
            ValueError: if there are no images in the ImageVolume

        """
        if self.PixelIntensityNormalized:
            return

        if self.ImageVolume is None or not len(self.ImageVolume):
            raise ValueError("No imported image volume to normalize")

        self.ImageVolume = [
            self._normalize_image_pixel_intensity_relationship(image, self.CompleteMetadata[ind])
            for ind, image in enumerate(self.ImageVolume)
        ]

        self.PixelIntensityNormalized = True

    @staticmethod
    def _normalize_image_pixel_intensity_relationship(image: np.ndarray, metadata: FileDataset) -> np.ndarray:
        if metadata.PixelIntensityRelationshipSign == 1:
            return image

        image = np.multiply(image - np.power(2, metadata.BitsStored), -1)

        return image
