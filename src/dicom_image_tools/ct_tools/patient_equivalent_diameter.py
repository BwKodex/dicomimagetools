import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from skimage.transform import hough_line, radon

from ..dicom_handlers.ct import CtSeries
from ..helpers.voxel_data import VoxelData

log = logging.getLogger(__name__)


@dataclass
class EquivalentDiameterData:
    Area_px: float
    EAD_px: float
    EAD_cm: float
    EquivalentAreaCircumference_cm: float
    MeanHU: float
    MedianHU: float
    LAT_cm: float
    AP_cm: float
    WED_px: float
    WED_cm: float
    EED_px: float
    EED_cm: float

    def to_dict(self):
        return {
            "Area_px": self.Area_px,
            "EAD_px": self.EAD_px,
            "EAD_cm": self.EAD_cm,
            "EquivalentAreaCircumference_cm": self.EquivalentAreaCircumference_cm,
            "MeanHU": self.MeanHU,
            "MedianHU": self.MedianHU,
            "LAT_cm": self.LAT_cm,
            "AP_cm": self.AP_cm,
            "WED_px": self.WED_px,
            "WED_cm": self.WED_cm,
            "EED_px": self.EED_px,
            "EED_cm": self.EED_cm,
        }

    def to_json(self):
        return json.dumps(self.to_dict())


@dataclass
class SinogramData:
    MaxPixels: float
    MinPixels: float
    MaxCm: float
    MinCm: float
    MaxAngle: float
    MinAngle: float
    EquivalentDiameter: float

    def to_dict(self):
        return {
            "MaxPixels": self.MaxPixels,
            "MinPixels": self.MinPixels,
            "MaxCm": self.MaxCm,
            "MinCm": self.MinCm,
            "MaxAngle": self.MaxAngle,
            "MinAngle": self.MinAngle,
        }

    def to_json(self):
        return json.dumps(self.to_dict())


def _calculate_slice_area_equivalent_diameter(
    image: np.ndarray, mask: np.ndarray, voxel_data: VoxelData
) -> EquivalentDiameterData:
    """Calculate the area equivalent diameter from a 2D image based on AAPM report 204 and 220. Return an
    EquivalentDiameterData instance.

    Args:
        image: A 2D numpy.ndarray containing the patient image
        mask: A 2D binary numpy.ndarray containing the patient mask/ROI
        voxel_data: Pixel size for the mask

    Returns:
        The patient area in pixels, equivalent area diameter (EAD), water equivalent diameter (WED), and elliptical
        equivalent diameter (EED) in pixels and cm, the equivalent area circumference in cm, mean and median
        CT-number/HU, and the lateral and anterio-posterior dimension in cm.

    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy ndarray")

    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy ndarray")

    if not isinstance(voxel_data, VoxelData):
        raise TypeError("voxel_data must be a VoxelData instance")

    im = image.copy()  # Prevent overwriting image

    mask_area_pixels = float(np.sum(mask))

    mask_equivalent_diameter_pixels = np.multiply(2.0, np.sqrt(np.divide(mask_area_pixels, np.pi)))
    mask_equivalent_diameter_cm = np.multiply(mask_equivalent_diameter_pixels, np.divide(voxel_data.x, 10.0))
    mask_equivalent_circumference_pixels = np.multiply(2.0, np.sqrt(np.multiply(mask_area_pixels, np.pi)))
    mask_equivalent_circumference_cm = np.multiply(mask_equivalent_circumference_pixels, np.divide(voxel_data.x, 10.0))

    mean_hu = float(np.mean(im[mask]))
    median_hu = float(np.median(im[mask]))

    # Aw = 1/1000 * AvgCT*Aroi + Aroi  from AAPM Report 220 §2.1
    # Aw = sum(CT(x,y))/1000 * Apixel + sum(Apixel)  from AAPM Report 220 §2.1
    # -> = sum(CT(x,y))/1000 * Apixel + Npixel*Apixel
    # -> = (sum(CT(x,y))/1000 + Npixel) * Apixel   # Apixel = area of pixel = pixelspacing[0]*pixelspacing[1]
    aw_pixels = np.add(np.multiply(np.divide(mean_hu, 1000.0), mask_area_pixels), mask_area_pixels)
    wed_pixels = np.multiply(2.0, np.sqrt(np.divide(aw_pixels, np.pi)))
    wed_cm = np.multiply(wed_pixels, np.divide(voxel_data.x, 10.0))

    # Calculate AP and LAT dimensions
    lat_pixels = np.count_nonzero(np.sum(mask, axis=0))
    lat_cm = np.multiply(lat_pixels, np.divide(voxel_data.x, 10.0))
    ap_pixels = np.count_nonzero(np.sum(mask, axis=1))
    ap_cm = np.multiply(ap_pixels, np.divide(voxel_data.y, 10.0))

    # Elliptical efficient diameter (EED) as suggested in AAPM report 204
    # Calculated from the AP and LAT dimensions
    eed_pixels = np.sqrt(np.multiply(ap_pixels, lat_pixels))
    eed_cm = np.sqrt(np.multiply(ap_cm, lat_cm))

    return EquivalentDiameterData(
        Area_px=mask_area_pixels,
        EAD_px=mask_equivalent_diameter_pixels,
        EAD_cm=mask_equivalent_diameter_cm,
        EquivalentAreaCircumference_cm=mask_equivalent_circumference_cm,
        MeanHU=mean_hu,
        MedianHU=median_hu,
        LAT_cm=lat_cm,
        AP_cm=ap_cm,
        WED_px=wed_pixels,
        WED_cm=wed_cm,
        EED_px=eed_pixels,
        EED_cm=eed_cm,
    )


def calculate_max_min_lat_ap_hough(mask: np.ndarray, voxel_data: VoxelData):
    """Calculate the maximum and minimum distance through the patient/mask, and the angles at which they occur, using
    the Hough transform.

    Args:
        mask: A 2D binary numpy.ndarray containing the patient mask/ROI
        voxel_data: Pixel size for the mask

    Returns:
        The maximum and minimum distance through the patient/mask in pixels and cm, and the angles of the maximum and
        minimum distance, respectively.
    """
    if not isinstance(mask, np.ndarray) or (len(mask.shape) > 2 and mask.shape[2] > 1):
        raise TypeError("mask must be a 2D numpy ndarray")

    if not isinstance(voxel_data, VoxelData):
        raise TypeError("voxel_data must be a VoxelData object instance")

    h, angle, f = hough_line(mask, np.linspace(0, np.pi - np.divide(np.pi, 180), 180))

    # Calculate number of non-zero elements for each Hough transform line
    non_zero = [np.count_nonzero(h[:, ind]) for ind in range(0, 180)]

    # Get the maximum size of the patient from the sinogram
    sinogram_max_px = max(non_zero)
    sinogram_max_cm = np.multiply(sinogram_max_px, np.divide(voxel_data.x, 10.0))  # Assumes max patient size = LAT

    # Get the angle of the sinogram_max_px
    max_indices = np.where(np.array(non_zero) == sinogram_max_px)
    min_diffs = [
        min(angle[max_indices]),
        min([np.absolute(ang - np.divide(np.pi, 2.0)) for ang in angle[max_indices]]),
        min([np.absolute(ang - np.pi) for ang in angle[max_indices]]),
    ]
    tmp_min_ind = min_diffs.index(min(min_diffs))
    if tmp_min_ind == 0:
        sinogram_max_degree = [ang for ang in angle[max_indices] if ang == min_diffs[tmp_min_ind]][0]
    elif tmp_min_ind == 1:
        sinogram_max_degree = [
            ang for ang in angle[max_indices] if np.absolute(ang - np.divide(np.pi, 2.0)) == min_diffs[tmp_min_ind]
        ][0]
    else:
        sinogram_max_degree = [ang for ang in angle[max_indices] if np.absolute(ang - np.pi) == min_diffs[tmp_min_ind]][
            0
        ]

    sinogram_max_degree = np.multiply(sinogram_max_degree, np.divide(180.0, np.pi))

    # Get the minimum size of the patient from the sinogram
    sinogram_min_px = min(non_zero)
    sinogram_min_cm = np.multiply(sinogram_min_px, np.divide(voxel_data.y, 10.0))  # Assumes min patient size = AP

    min_indices = np.where(np.array(non_zero) == sinogram_min_px)
    min_diffs = [
        min(angle[min_indices]),
        min([np.absolute(ang - np.divide(np.pi, 2.0)) for ang in angle[min_indices]]),
        min([np.absolute(ang - np.pi) for ang in angle[min_indices]]),
    ]
    tmp_min_ind = min_diffs.index(min(min_diffs))
    if tmp_min_ind == 0:
        sinogram_min_degree = [ang for ang in angle[min_indices] if ang == min_diffs[tmp_min_ind]][0]
    elif tmp_min_ind == 1:
        sinogram_min_degree = [
            ang for ang in angle[min_indices] if np.absolute(ang - np.divide(np.pi, 2.0)) == min_diffs[tmp_min_ind]
        ][0]
    else:
        sinogram_min_degree = [ang for ang in angle[min_indices] if np.absolute(ang - np.pi) == min_diffs[tmp_min_ind]][
            0
        ]
    sinogram_min_degree = np.multiply(sinogram_min_degree, np.divide(180.0, np.pi))

    return SinogramData(
        MaxPixels=sinogram_max_px,
        MaxCm=sinogram_max_cm,
        MaxAngle=sinogram_max_degree,
        MinPixels=sinogram_min_px,
        MinCm=sinogram_min_cm,
        MinAngle=sinogram_min_degree,
        EquivalentDiameter=np.sqrt(np.multiply(sinogram_max_cm, sinogram_min_cm)),
    )


def calculate_max_min_lat_ap_radon(mask: np.ndarray, voxel_data: VoxelData) -> SinogramData:
    """Calculate the maximum and minimum distance through the patient/mask, and the angles at which they occur, using
    the radon transform.

    Args:
        mask: A 2D binary numpy.ndarray containing the patient mask/ROI
        voxel_data: Pixel size for the mask

    Returns:
        The maximum and minimum distance through the patient/mask in pixels and cm, and the angles of the maximum and
        minimum distance, respectively.
    """
    if not isinstance(mask, np.ndarray) or (len(mask.shape) > 2 and mask.shape[2] > 1):
        raise TypeError("mask must be a 2D numpy ndarray")

    if not isinstance(voxel_data, VoxelData):
        raise TypeError("voxel_data must be a VoxelData object instance")

    radon_mask = radon(mask, theta=np.arange(180), circle=True, preserve_range=True)

    non_zero = [np.count_nonzero(radon_mask[:, ind]) for ind in range(180)]

    sinogram_max_px = max(non_zero)
    sinogram_max_cm = np.multiply(sinogram_max_px, np.divide(voxel_data.x, 10.0))  # Assumes max patient size = LAT

    # Get the angle of the sinogram_max_px
    max_indices = list(np.where(np.array(non_zero) == sinogram_max_px)[0])
    min_diffs = [
        min(max_indices),
        min([np.absolute(ang - 90.0) for ang in max_indices]),
        min([np.absolute(ang - 180.0) for ang in max_indices]),
    ]
    tmp_min_ind = min_diffs.index(min(min_diffs))
    if tmp_min_ind == 0:
        sinogram_max_degree = float([ang for ang in max_indices if ang == min_diffs[tmp_min_ind]][0])
    elif tmp_min_ind == 1:
        sinogram_max_degree = float(
            [ang for ang in max_indices if np.absolute(ang - 90.0) == min_diffs[tmp_min_ind]][0]
        )
    else:
        sinogram_max_degree = float(
            [ang for ang in max_indices if np.absolute(ang - 180.0) == min_diffs[tmp_min_ind]][0]
        )

    sinogram_min_px = min(non_zero)
    sinogram_min_cm = np.multiply(sinogram_min_px, np.divide(voxel_data.y, 10.0))  # Assumes min patient size = AP

    min_indices = list(np.where(np.array(non_zero) == sinogram_min_px)[0])
    min_diffs = [
        min(min_indices),
        min([np.absolute(ang - 90.0) for ang in min_indices]),
        min([np.absolute(ang - 180.0) for ang in min_indices]),
    ]
    tmp_min_ind = min_diffs.index(min(min_diffs))
    if tmp_min_ind == 0:
        sinogram_min_degree = float([ang for ang in min_indices if ang == min_diffs[tmp_min_ind]][0])
    elif tmp_min_ind == 1:
        sinogram_min_degree = float(
            [ang for ang in min_indices if np.absolute(ang - 90.0) == min_diffs[tmp_min_ind]][0]
        )
    else:
        sinogram_min_degree = float(
            [ang for ang in min_indices if np.absolute(ang - 180.0) == min_diffs[tmp_min_ind]][0]
        )

    return SinogramData(
        MaxPixels=sinogram_max_px,
        MaxCm=sinogram_max_cm,
        MaxAngle=sinogram_max_degree,
        MinPixels=sinogram_min_px,
        MinCm=sinogram_min_cm,
        MinAngle=sinogram_min_degree,
        EquivalentDiameter=np.sqrt(np.multiply(sinogram_max_cm, sinogram_min_cm)),
    )


def calculate_area_equivalent_diameter(
    ct: CtSeries, use_radon: Optional[bool] = False, use_hough: Optional[bool] = False
) -> Tuple[List[EquivalentDiameterData], Optional[List[SinogramData]], Optional[List[SinogramData]]]:
    """Calculate the area equivalent diameter for all slices in a CT series. The calculations are based on the
    specifications in AAPM report 204 and 220.

    Args:
        ct: A CtSeries object containing the series for which the area equivalent diameter should be determined
        use_radon: Use radon transform to determine maximum and minimum distance through patient
        use_hough: Use Hough transform to determine maximum and minimum distance through patient

    Returns:
        Tuple in the form (EAD_data, radon_data, hough_data) where the radon_data and hough_data only have values if
        use_radon and/or use_hough was set to true, respectively.
    """
    if not isinstance(ct, CtSeries):
        raise TypeError("ct must be an instance of the CtSeries class")

    if not isinstance(use_radon, bool):
        raise TypeError("use_radon must be a boolean")

    if not isinstance(use_hough, bool):
        raise TypeError("use_hough must be a boolean")

    log.debug("Calculating area equivalent diameter")
    if ct.Mask is None:
        ct.get_patient_mask(remove_table=True)

    ead_data = [
        _calculate_slice_area_equivalent_diameter(ct.ImageVolume[:, :, ind], ct.Mask[:, :, ind], ct.VoxelData[ind])
        for ind in range(ct.ImageVolume.shape[2])
    ]

    radon_data = None
    if use_radon:
        radon_data = [
            calculate_max_min_lat_ap_radon(mask=ct.Mask[:, :, ind], voxel_data=ct.VoxelData[ind])
            for ind in range(ct.ImageVolume.shape[2])
        ]

    hough_data = None
    if use_hough:
        hough_data = [
            calculate_max_min_lat_ap_radon(mask=ct.Mask[:, :, ind], voxel_data=ct.VoxelData[ind])
            for ind in range(ct.ImageVolume.shape[2])
        ]

    return ead_data, radon_data, hough_data
