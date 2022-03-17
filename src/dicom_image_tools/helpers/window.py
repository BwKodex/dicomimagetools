import logging
from typing import Optional

import numpy as np
from pydicom import FileDataset

logger = logging.getLogger(__name__)


def get_default_window_settings(
    metadata: FileDataset, image_slice: np.ndarray, modality: Optional[str] = None
) -> tuple[float, float]:
    if "WindowCenter" in metadata and "WindowWidth" in metadata:
        try:
            window_center = float(wc[0]) if isinstance((wc := metadata.WindowCenter), list) else float(wc)
            window_width = float(ww[0]) if isinstance((ww := metadata.WindowWidth), list) else float(ww)

            if isinstance(window_width, list):
                window_width = float(window_width[0])

            return (
                float(window_center - np.floor(window_width / 2)),
                float(window_center + np.ceil(window_width / 2)),
            )
        except:
            logger.error("Failed to extract window from metadata", exc_info=True)

    if modality is not None and modality.casefold() == "ct".casefold():
        return -500.0, 500.0

    return (float(np.min(image_slice)), float(np.max(image_slice)))
