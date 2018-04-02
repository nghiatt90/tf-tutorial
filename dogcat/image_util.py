import numpy as np
from skimage import io, transform, img_as_ubyte
from typing import Tuple, Union


def center_crop(image: np.ndarray) -> np.ndarray:
    """Center crop an image"""
    height, width, _ = image.shape
    new_height = new_width = min(height, width)
    return image[
               (height - new_height) // 2: (height - new_height) // 2 + new_height,
               (width - new_width) // 2: (width - new_width) // 2 + new_width,
               :
           ]


def read_and_resize(path: str, size: Union[Tuple[int, int], int] = None, do_center_crop: bool = False) -> np.ndarray:
    """Read and optionally resize an image."""
    assert isinstance(size, tuple) or isinstance(size, int) or size is None, 'Size must be Tuple[int, int], int or None'

    image = io.imread(path)

    if isinstance(size, int):
        do_center_crop = True

    if do_center_crop:
        image = center_crop(image)

    if size is not None:
        if isinstance(size, int):
            size = (size, size)
        image = transform.resize(image, size)
    return img_as_ubyte(image)
