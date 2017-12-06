import cv2
import numpy as np
import tensorflow as tf
from typing import ByteString


def int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value: ByteString) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_image(path: str, image_size: int = 299) -> np.ndarray:
    """
    Read an image and resize it for training/testing.

    :param path: Path to image
    :param image_size: Desirable size for training or testing
    :return: Image after resizing
    """
    image = cv2.imread(path)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = image.astype(np.float32)
    return image


def create_example_proto(path: str, label: int = None) -> tf.train.Example:
    """
    Create an instance of tf.train.Example from an image for training or testing.

    :param path: Path to the image
    :param label: Class label of the image
    :return: tf.train.Example
    """
    image = load_image(path)
    features = {
        'label': int64_feature(label),
        'image': bytes_feature(tf.compat.as_bytes(image.tostring()))
    } if label is not None else {
        'image': bytes_feature(tf.compat.as_bytes(image.tostring()))
    }
    return tf.train.Example(features=tf.train.Features(feature=features))
