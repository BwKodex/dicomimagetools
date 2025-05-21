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
        return float(tmp) if (tmp := dcm.XRayTubeCurrent) else tmp

    if "XRayTubeCurrentInmA" in dcm:
        return float(tmp) if (tmp := dcm.XRayTubeCurrentInmA) else tmp

    if "XRayTubeCurrentInuA" in dcm:
        return (float(tmp) / 1000) if (tmp := dcm.XRayTubeCurrentInuA) else tmp
