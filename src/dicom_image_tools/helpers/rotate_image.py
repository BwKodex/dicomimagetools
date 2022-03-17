import logging

import numpy as np
from pydicom import FileDataset
from scipy import ndimage

logger = logging.getLogger(__name__)


def rotate_image(image: np.ndarray, metadata: FileDataset) -> np.ndarray:
    """Rotates image to rotation 0 based on value in the FieldOfViewRotation data

    Args:
        image: Image to rotate given as an numpy.ndarray
        metadata:

    Returns:
        Rotated image as a numpy.ndarray

    """
    if "FieldOfViewRotation" not in metadata:
        raise ValueError("No field of view rotation data in the given metadata")

    rot_angle = metadata.FieldOfViewRotation

    if rot_angle % 90 == 0:
        return np.rot90(image, k=(rot_angle // 90))

    theta = np.deg2rad(-rot_angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    return _apply_rotation_matrix(image=image, rotation_matrix=rotation_matrix)


def _apply_rotation_matrix(
    image: np.ndarray,
    rotation_matrix: np.array,
    row_axis: int = 0,
    col_axis: int = 1,
    channel_axis: int = 2,
    fill_mode: str = "nearest",
    order: int = 1,
    value_outside_boundaries_of_input: float = 0,
) -> np.ndarray:
    offset_x = float(image.shape[row_axis]) / 2 + 0.5
    offset_y = float(image.shape[col_axis]) / 2 + 0.5
    offset_matrix = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -offset_x], [0, 1, -offset_y], [0, 0, 1]])
    rotation_matrix = np.dot(np.dot(offset_matrix, rotation_matrix), reset_matrix)

    image = np.rollaxis(image, channel_axis, 0)
    final_affine_matrix = rotation_matrix[:2, :2]
    final_offset = rotation_matrix[:2, 2]

    channel_images = [
        ndimage.interpolation.affine_transform(
            image_channel,
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=value_outside_boundaries_of_input,
        )
        for image_channel in image
    ]
    image = np.stack(channel_images, axis=0)
    image = np.rollaxis(image, 0, channel_axis + 1)

    return image
