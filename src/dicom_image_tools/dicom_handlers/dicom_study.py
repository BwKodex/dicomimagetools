import logging
import numpy as np
from pathlib import Path
import pydicom
from pydicom import FileDataset
from typing import List, Optional

from ..helpers.voxel_data import VoxelData


log = logging.getLogger(__name__)


class DicomSeries:
    """ A class to manage DICOM files connected by a Series Instance UID

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
        self.CompleteMetadata: List[FileDataset] = []
        self.VoxelData: List[VoxelData] = []

        self.ImageVolume: Optional[np.ndarray] = None
        self.Mask: Optional[np.ndarray] = None

    def add_file(self, file: Path, dcm: Optional[FileDataset] = None):
        """ Add a file to the objects list of files

        First performs a check that the file is a valid DICOM file and that it belongs to the object/series

        Args:
            file: Path to where the file to be added is stored on disk
            dcm: The DICOM-file imported to a FileDataset object

        Raises:
            ValueError: if SeriesInstanceUID of the file is not the same as the SeriesInstanceUid attribute

        """
        if dcm is None:
            dcm = pydicom.dcmread(fp=str(file.absolute()), stop_before_pixels=True)

        if any([True if obj == file else False for obj in self.FilePaths]):
            log.info('File already in object')
            return

        if dcm.SeriesInstanceUID != self.SeriesInstanceUid:
            msg = f"Wrong SeriesInstanceUID. Expected: {self.SeriesInstanceUid}; Input: {dcm.SeriesInstanceUID}"
            log.error(msg=msg)
            raise ValueError(msg)

        self.FilePaths.append(file)


class DicomStudy:
    """ A class to manage DICOM files connected by a Study Instance UID

    Args:
        study_instance_uid : The study instance UID of the DICOM study object that is to be created

    Attributes:
        StudyInstanceUid: The study instance UID of the DICOM study object
        Series: List of DicomSeries objects containing detailed information abouts series included in the study
        Manufacturer: The manufacturer of the machine used in acquiring the image/-s
        ManufacturerModelName: The name of the machine, as given by the manufacturer, used in acquiring the image/-s

    Raises:
        TypeError: if study_instance_uid is not a string

    """
    def __init__(self, study_instance_uid: str):
        if not isinstance(study_instance_uid, str):
            raise TypeError("study_instance_uid must be a string")
        # Metadata
        self.StudyInstanceUid: str = study_instance_uid
        self.Series: List[DicomSeries] = []
        self.Manufacturer: Optional[str] = None
        self.ManufacturerModelName: Optional[str] = None

    def add_file(self, file: Path, dcm: Optional[FileDataset] = None) -> None:
        """ Add the DICOM file to the DicomStudy object after validating the study instance UID

        Args:
            file: Path to where the file to be added is stored on disk
            dcm: The DICOM-file imported to a FileDataset object

        Raises:
            InvalidDicomError: If the given file is not a valid DICOM file
            ValueError: If the file does not have the same study instance UID as the StudyInstanceUID of the object

        """
        if dcm is None:
            dcm = pydicom.dcmread(fp=str(file.absolute()), stop_before_pixels=True)

        if dcm.StudyInstanceUID != self.StudyInstanceUid:
            raise ValueError(f"The given DICOM file is not part of the study {self.StudyInstanceUid}")

        self.Manufacturer = dcm.Manufacturer
        self.ManufacturerModelName = dcm.ManufacturerModelName

        try:
            index = [obj.SeriesInstanceUid for obj in self.Series].index(dcm.SeriesInstanceUID)
        except ValueError as e:
            log.debug(f"Found new Series {dcm.SeriesInstanceUID}")
            self.Series.append(DicomSeries(dcm.SeriesInstanceUID))
            index = -1

        self.Series[index].add_file(file=file, dcm=dcm)
        log.debug("File added to object")
