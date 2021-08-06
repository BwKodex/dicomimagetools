import numpy as np
from pydicom.dataset import Dataset, FileDataset
from pytest import fixture


@fixture(scope="function")
def image_with_negative_pixel_intensity_relationship():
    ds: Dataset = Dataset()
    ds.PixelIntensityRelationshipSign = -1
    ds.BitsStored = 12

    return {
        "image": np.array([[0, 0, 0], [2 ** 10, 2 ** 10, 2 ** 10]]),
        "normalized_image": np.multiply(
            np.array(
                [
                    [-(2 ** ds.BitsStored), -(2 ** ds.BitsStored), -(2 ** ds.BitsStored)],
                    [2 ** 10 - 2 ** ds.BitsStored, 2 ** 10 - 2 ** ds.BitsStored, 2 ** 10 - 2 ** ds.BitsStored],
                ]
            ),
            -1,
        ),
        "metadata": ds,
    }
