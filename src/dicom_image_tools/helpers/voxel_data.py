from dataclasses import dataclass
from typing import Optional


@dataclass
class VoxelData:
    """A class for managing voxel/pixel data for DICOM images

    Attributes:
        x: The voxel/pixel x-dimension in mm
        y: The voxel/pixel y-dimension in mm
        z: The voxel/pixel z-dimension in mm
    """

    x: float
    y: float
    z: Optional[float] = None
    unit: Optional[str] = "mm"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z and self.unit == other.unit

    def pixel_area(self) -> float:
        return self.x * self.y

    def volume(self) -> float:
        return self.x * self.y * self.z
