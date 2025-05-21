from pathlib import Path
from typing import Union

import numpy as np
import pydicom


def save_dicom(image: np.ndarray, metadata: pydicom.FileDataset, output_path: Union[Path, str], bits_allocated: int = 16):
    if isinstance(output_path, str):
        output_path = Path(output_path)

    if not isinstance(output_path, Path):
        raise TypeError("The output path must be a Path or a string object")

    if not output_path.parent.exists():
        raise ValueError("The output directory does not exist")

    max_val = image.max()
    min_val = image.min()

    value_range = max_val - min_val

    metadata.RescaleIntercept = "0"
    metadata.RescaleSlope = "1"

    if min_val < 0 or max_val >= 2 ** bits_allocated:
        metadata.RescaleIntercept = str(int(min_val))
        image = image - min_val

        if value_range >= 2 ** bits_allocated:
            rescale_factor = value_range / (2 ** bits_allocated - 1)
            image = image / rescale_factor
            metadata.RescaleSlope = rescale_factor

    metadata.BitsAllocated = bits_allocated
    metadata.BitsStored = bits_allocated
    metadata.Rows = image.shape[0]
    metadata.Columns = image.shape[1]

    image = (np.rint(image)).astype(np.int16)
    metadata.PixelData = image.tobytes()
    metadata["PixelData"].VR = "OW"

    metadata.is_little_endian = True
    metadata.is_implicit_VR = False

    metadata.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    pydicom.dcmwrite(output_path, metadata)
