from typing import Optional, Union

from pydicom import Dataset, FileDataset


def get_xray_tube_current_in_ma(dcm: Union[FileDataset, Dataset]) -> Optional[float]:
    """Checks the DICOM dataset for the X-ray tube current in a set of different tags. If a value for the tube current
    is found, it is converted to mA unit and returned

    Args:
        dcm: The DICOM image dataset

    Returns:
        The tube current converted to mA
    """
    if "XRayTubeCurrent" in dcm:
        return float(dcm.XRayTubeCurrent)

    if "XRayTubeCurrentInmA" in dcm:
        return float(dcm.XRayTubeCurrentInmA)

    if "XRayTubeCurrentInuA" in dcm:
        return float(dcm.XRayTubeCurrentInuA) / 1000
