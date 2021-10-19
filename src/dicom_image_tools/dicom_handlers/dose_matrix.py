import logging
from pathlib import Path
from typing import Optional, List, Union

import numpy as np
import pydicom
from pydicom import FileDataset

from .dicom_series import DicomSeries
from ..helpers.check_path_is_valid import check_path_is_valid_path
from ..helpers.pixel_data import get_pixel_array
from ..helpers.voxel_data import VoxelData

logger = logging.getLogger(__name__)


class DoseMatrix(DicomSeries):
    """A class to manage dose matrix series from Treatment planning systems.

    """
    def __init__(self, file: Path, dcm: Optional[FileDataset] = None):
        if dcm is None:
            dcm = pydicom.dcmread(fp=file, stop_before_pixels=True)

        if "SeriesInstanceUID" not in dcm:
            raise ValueError("The DICOM file does not contain a series instance UID")

        super().__init__(series_instance_uid=dcm.SeriesInstanceUID)
        self.DoseGridScaling: Optional[List[Optional[float]]] = []
        self.DoseSummationType: Optional[List[Optional[str]]] = []
        self.DoseType: Optional[List[Optional[str]]] = []
        self.DoseUnit: Optional[List[Optional[str]]] = []
        self.ImageVolume: Optional[List[np.ndarray]] = []
        self.Origin: Optional[List[List[float]]] = []

        self.add_file(file=file, dcm=dcm)

    def add_file(self, file: Union[Path, str], dcm: Optional[FileDataset] = None):
        """Add a file to the objects list of files

        First performs a check that the file path is valid and that is has the same Series Instance UID as the class
        object

        Args:
            file: Path to where the file to be added is stored on disc
            dcm: The DICOM-file imported to a FileDataset object

        Raises:
            TypeError: If the file is not a valid path
            ValueError: if SeriesInstanceUID of the file is not the same as the SeriesInstanceUid attribute
        """
        file = check_path_is_valid_path(file)

        if any([True if obj == file else False for obj in self.FilePaths]):
            # Return None since the file is already in the volume
            return

        if dcm is None:
            dcm = pydicom.dcmread(fp=str(file.absolute()), stop_before_pixels=True)

        if dcm.SeriesInstanceUID != self.SeriesInstanceUid:
            msg = f"Wrong SeriesInstanceUID. Expected: {self.SeriesInstanceUid}; Input: {dcm.SeriesInstanceUID}"
            raise ValueError(msg)

        self.DoseGridScaling.append(float(dcm.DoseGridScaling) if "DoseGridScaling" in dcm else None)
        self.DoseSummationType.append(dcm.DoseSummationType if "DoseSummationType" in dcm else None)
        self.DoseType.append(dcm.DoseType if "DoseType" in dcm else None)
        self.DoseUnit.append(dcm.DoseUnit if "DoseUnit" in dcm else None)
        self.Origin.append([float(pos) for pos in dcm.ImagePositionPatient] if "ImagePositionPatient" in dcm else None)
        self.FilePaths.append(file)

        if "PixelSpacing" in dcm:
            try:
                self.VoxelData.append(
                    VoxelData(x=float(dcm.PixelSpacing[1]), y=float(dcm.PixelSpacing[0]), z=float(dcm.SliceThickness))
                )
            except:
                logger.error("Failed to set VoxelData for dose matrix", exc_info=True)
                self.VoxelData.append(None)

        # Remove pixel data part of dcm to decrease memory used for the object
        if "PixelData" in dcm:
            try:
                del dcm[0x7FE00010]
            except Exception:
                logger.debug(
                    "Failed to remove pixel data from file before appending to CompleteMetadata", exc_info=True
                )
                pass

        self.CompleteMetadata.append(dcm)

    def import_image_volume(self) -> None:
        """Import the file/-s in the Dose matrix volume and insert the data into the ImageVolume property

        Returns:

        """
        for ind, fp in enumerate(self.FilePaths):
            dcm = pydicom.dcmread(fp)

            self.ImageVolume.append(get_pixel_array(dcm=dcm))

    def import_dose_matrix(self) -> None:
        """Import the file/-s in the Dose matrix volume and insert the data into the ImageVolume property

        See :func:`~dicom_image_tools.dicom_handlers.dose_matrix.DoseMatrix.import_image_volume`
        """
        self.import_image_volume()
