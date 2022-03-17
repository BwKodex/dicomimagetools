from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pydicom
from plotly import graph_objects as go
from pydicom import FileDataset

from dicom_image_tools.plotting.colour_scales import plotly_colour_scales

from ..helpers.check_path_is_valid import check_path_is_valid_path
from ..helpers.voxel_data import VoxelData
from ..plotting.plotly import show_image
from ..roi.roi import Roi


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

    def add_file(self, file: Union[Path, str], dcm: Optional[FileDataset] = None):
        """Add a file to the objects list of files

        First performs a check that the file is a valid DICOM file and that it belongs to the object/series

        Args:
            file: Path to where the file to be added is stored on disk
            dcm: The DICOM-file imported to a FileDataset object

        Raises:
            ValueError: if SeriesInstanceUID of the file is not the same as the SeriesInstanceUid attribute
            TypeError: if file is not a valid/existing path

        """
        file = check_path_is_valid_path(path_to_check=file)

        if any([True if obj == file else False for obj in self.FilePaths]):
            # Return None since the file is already in the volume
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

    def show_image(
        self,
        index: int = 0,
        rois: Optional[list[Roi]] = None,
        colour_map: str = "bone",
        window: Optional[tuple[float, float]] = None,
    ) -> go.Figure:
        if not isinstance(index, int):
            raise TypeError("Image index must be given as an integer")

        if index < 0 or index > (len(self.FilePaths) - 1):
            raise ValueError("Invalid image index specified")

        if rois is not None and (not isinstance(rois, list) or not all([isinstance(roi, list) for roi in rois])):
            raise TypeError("Only list of SquareRoi instances are implemented for plotting")

        ALLOWED_COLOUR_MAPS = list(plotly_colour_scales.keys())

        if not isinstance(colour_map, str):
            raise TypeError(
                f"The colour map must be given as the name of an implemented colour map. Allowed values are {', '.join(ALLOWED_COLOUR_MAPS)}"
            )

        if colour_map not in ALLOWED_COLOUR_MAPS:
            raise NotImplementedError(
                f"Colour map must be one of the following implemented colour maps: {', '.join(ALLOWED_COLOUR_MAPS)}"
            )

        if window is not None:
            if not isinstance(window, (tuple, list)):
                raise TypeError("If specified, the window must be a tuple of length 2")

            if len(window) != 2 or any([not isinstance(val, (float, int)) for val in window]):
                raise ValueError("The specified window must be a tuple of exactly 2 floats")
