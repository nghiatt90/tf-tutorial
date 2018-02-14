import tensorflow as tf
from typing import ByteString

from image_util import read_and_resize


def int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value: ByteString) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_example_proto(path: str, label: int = None) -> tf.train.Example:
    """
    Create an instance of tf.train.Example from an image for training or testing.

    :param path: Path to the image
    :param label: Class label of the image
    :return: tf.train.Example
    """
    image = read_and_resize(path, size=299)
    features = {
        'label': int64_feature(label),
        'image': bytes_feature(tf.compat.as_bytes(image.tostring()))
    } if label is not None else {
        'image': bytes_feature(tf.compat.as_bytes(image.tostring()))
    }
    return tf.train.Example(features=tf.train.Features(feature=features))
