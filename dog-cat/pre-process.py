import cv2
import glob
import os
import numpy as np
import tensorflow as tf


def create_tfrecords(data_path: str) -> None:
    """
    Create TensorFlow-friendly TFRecord files from raw image data.
    Ref: http://machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html

    :param data_path: Path to a extracted training data
    :return: None
    """
    if False:
        print('Found TFRecord files. Skip converting step.')
        return None
    print('Converting raw images to TFRecords')
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

