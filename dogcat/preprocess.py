import cv2
import glob
import os
import numpy as np
import sys
import tensorflow as tf
from typing import ByteString, List, Tuple


# Ref: http://machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
# Ref: https://github.com/kwotsin/create_tfrecords/blob/master/dataset_utils.py

SHARD_COUNT = {
    'train': 15,
    'val': 5,
    'test': 5
}
TFRECORD_DIR_NAME = 'tfrecords'


def create_tfrecords(data_path: str) -> str:
    """
    Create TensorFlow-friendly TFRecord files from raw image data.

    :param data_path: Path to a extracted training data
    :return: None
    """
    output_dir = os.path.join(data_path, TFRECORD_DIR_NAME)
    if os.path.exists(output_dir):
        print('Found TFRecord directory. Skip converting step.')
        print('To re-run this step, delete directory %s under %s' % (TFRECORD_DIR_NAME, data_path))
    else:
        print('Converting raw images to TFRecords')
        datasets = shuffle_and_divide(data_path)
        assert len(datasets) == 3, 'Datasets must contain 3 tuples for train, validation and test sets'
        split_names = ('train', 'val', 'test')
        for (paths, labels), split_name in zip(datasets, split_names):
            convert_to_tfrecords(output_dir, split_name, paths, labels)

    return output_dir


def shuffle_and_divide(data_path: str) -> Tuple[Tuple[List[str], List[int]],
                                                Tuple[List[str], List[int]],
                                                Tuple[List[str], List[int]]]:
    """
    Shuffle and divide all images into train, validation and test sets.
    Images are not actually read here.

    :param data_path: Path to image directory
    :return:
    """
    image_pattern = os.path.join(data_path, '*.jpg')
    image_paths = glob.glob(image_pattern)
    labels = [0 if 'cat.' in image_path else 1 for image_path in image_paths]  # cat = 0, dog = 1

    # Shuffle data
    pairs = list(zip(image_paths, labels))
    np.random.shuffle(pairs)
    image_paths, labels = zip(*pairs)

    # Divide data into 60% train, 20% validation and 20% test
    total = len(image_paths)
    train_paths = image_paths[:int(0.6 * total)]
    train_labels = labels[:int(0.6 * total)]

    val_paths = image_paths[int(0.6 * total):int(0.8 * total)]
    val_labels = labels[int(0.6 * total):int(0.8 * total)]

    test_paths = image_paths[int(0.8 * total):]
    test_labels = labels[int(0.8 * total):]

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


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


def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value: ByteString) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _create_example_proto(path: str, label: int) -> tf.train.Example:
    image = load_image(path)
    features = {
        'label': _int64_feature(label),
        'image': _bytes_feature(tf.compat.as_bytes(image.tostring()))
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def convert_to_tfrecords(output_dir: str,
                         split_name: str, paths: List[str], labels: List[int],
                         shard_count: int = None) -> None:
    """
    Convert a dataset into tfrecord shards.

    :param output_dir: Directory to save output shards
    :param split_name: One of 'train', 'val', or 'test'
    :param paths: List of image paths
    :param labels: Images labels
    :param shard_count: Number of shards
    :return:
    """
    assert split_name in SHARD_COUNT.keys()
    assert len(paths) == len(labels)
    if shard_count is None or shard_count <= 0:
        shard_count = SHARD_COUNT[split_name]
    assert shard_count < len(paths), 'Too many shards'

    total = len(paths)
    images_per_shard = int(np.ceil(total / shard_count))

    # Prepare output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for shard_id in range(shard_count):
        output_name = '%s-%05d-of-%05d.tfrecord' % (split_name, shard_id + 1, shard_count)
        output_path = os.path.join(output_dir, output_name)
        with tf.python_io.TFRecordWriter(output_path) as writer:
            start_idx = shard_id * images_per_shard
            end_idx = min(start_idx + images_per_shard, total)
            for i in range(start_idx, end_idx):
                sys.stdout.write('\rConverting image %d/%d shard %d' % (i + 1, total, shard_id + 1))
                sys.stdout.flush()
                example_proto = _create_example_proto(paths[i], labels[i])
                writer.write(example_proto.SerializeToString())

        sys.stdout.write('\n')
        sys.stdout.flush()
