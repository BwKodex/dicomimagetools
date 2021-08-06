from pathlib import Path

import pytest

from dicom_image_tools.dicom_handlers.dicom_import import import_dicom_file
from dicom_image_tools.helpers.point import Point
from dicom_image_tools.image_quality.mtf import mtf_edge_phantom
from dicom_image_tools.roi.square_roi import SquareRoi


def test_mtf_from_edge_phantom():
    file = Path(__file__).parent.parent / "test_data" / "io" / "Edge-fantom" / "Image9.dcm"

    im = import_dicom_file(file=file)
    im_shape = im.Series[0].ImageVolume[0].shape
    im_roi = SquareRoi(
        center=Point(x=im_shape[1] / 2, y=im_shape[0] / 2),
        height=im_shape[0] * 0.1 * im.Series[0].VoxelData[0].y,
        width=im_shape[1] * 0.4 * im.Series[0].VoxelData[0].x,
        pixel_size=im.Series[0].VoxelData[0],
    )

    mtf = mtf_edge_phantom(image=im.Series[0].ImageVolume[0], roi=im_roi)

    assert max(mtf.mtf) == 1.0
    assert mtf.mtf[12] < 0.5
