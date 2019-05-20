from dataclasses import dataclass
from typing import Optional


@dataclass
class VoxelData:
    """
    A class for managing voxel/pixel data for DICOM images

    Attributes
    ----------
    x : float
        The voxel/pixel x-dimension in mm
    y : float
        The voxel/pixel y-dimension in mm
    z : Optional[float]
        The voxel/pixel z-dimension in mm
    """
    x: float
    y: float
    z: Optional[float]
