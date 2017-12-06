import cv2
import numpy as np
import tensorflow as tf
from typing import ByteString


def int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value: ByteString) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_image(path: str) -> np.ndarray:
    """
    Read an image, resize to 224x224.

    :param path: Path to image
    :return: Image after resizing
    """
    image = cv2.imread(path)
    image = cv2.resize(image, (299, 299), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = image.astype(np.float32)
    return image
