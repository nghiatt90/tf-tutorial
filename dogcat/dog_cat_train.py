import argparse
import configparser
import glob
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.data import TFRecordDataset
from tensorflow import keras
import time
from typing import Dict, List, Tuple

from preprocess import create_tfrecords


tf.logging.set_verbosity(tf.logging.INFO)

CONFIG_SECTION_INPUT = 'input'
CONFIG_SECTION_OUTPUT = 'output'
CONFIG_SECTION_HYPERPARAMS = 'hyperparameters'
IMAGENET_WEIGHTS = 'imagenet'
OUTPUT_CLASS_COUNT = 2


# noinspection PyShadowingNames
def validate_input(args: argparse.Namespace) -> None:
    """Validate user input"""

    # args.data must be a valid directory
    data_path = args.data_dir
    assert os.path.exists(data_path), 'Cannot find data directory: %s' % data_path
    assert os.path.isdir(data_path), '%s is not a directory' % data_path


def get_tfrecord_files(data_dir: str, split_name: str) -> List[str]:
    """Get a list of tfrecord files corresponding to split_name.

    :param data_dir: Path to data containing directory
    :param split_name: One of 'train', 'val', 'test'
    :return: List of tfrecord file names (string)
    """
    pattern = os.path.join(data_dir, '%s-*.tfrecord' % split_name)
    return glob.glob(pattern)


def get_tfrecord_loader(file_names: List[str],
                        batch_size: int = None, buffer_size: int = 256, epochs: int = None,
                        shuffle: bool = True) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Create a dataset to read tfrecord files and return its iterator.

    The iterator expects a list of tfrecord file names to be fed to
    its 'file_names' placeholder.

    :param file_names: tf.placeholder
    :param batch_size: Number of images in each batch
    :param buffer_size: See tf.contrib.Dataset.shuffle
    :param epochs:
    :param shuffle: Whether or not to shuffle the dataset
    :return: Tensor of type Iterator
    """

    feature_map = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }

    def parse_example_proto(proto: tf.train.Example, image_size: int = 299, channels: int = 3) \
            -> Tuple[Dict[str, tf.Tensor], List[tf.Tensor]]:
        """

        :param proto:
        :param image_size:
        :param channels:
        :return:
        """
        features = tf.parse_single_example(proto, features=feature_map)
        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.reshape(image, [image_size, image_size, channels])
        image = tf.cast(image, tf.float32)
        image = tf.subtract(image, 116.779)
        label = tf.cast(features['label'], tf.float32)
        return dict(zip(['input_1'], [image])), [label]

    dataset = TFRecordDataset(file_names)
    dataset = dataset.map(parse_example_proto)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    dataset.repeat(epochs)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    return images, labels


def build_model(weights: str = None, lr: float = 1e-3, momentum: float = 0.9,
                model_dir: str = None) -> tf.estimator.Estimator:
    # Use Keras's Inception v3 model without weights to retrain from scratch.
    # include_top = False removes the last fully connected layer.
    base_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights=weights)

    # Replace last layer with our own.
    # GlobalAveragePooling2D converts the MxNxC tensor output into a 1xC tensor where C is the # of channels.
    # Dense is a fully connected layer.
    layers = keras.layers
    model = keras.models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    base_model.trainable = False

    model.compile(
        optimizer=keras.optimizers.RMSprop(lr=lr),
        loss='binary_crossentropy',
        metric=['accuracy']
    )
    # print(model.summary())

    model_dir = os.path.join(os.getcwd(), model_dir)
    os.makedirs(model_dir, exist_ok=True)
    return keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir)


# noinspection PyShadowingNames
def train(args: argparse.Namespace) -> None:
    """
    Train a new model.

    :param args: Validated user inputs
    :return: None
    """
    start_time = time.time()
    input_dir = create_tfrecords(args.data_dir)
    checkpoint1 = time.time()
    model = build_model(IMAGENET_WEIGHTS, args.learning_rate, args.momentum, args.output_dir)
    checkpoint2 = time.time()

    train_file_names = get_tfrecord_files(input_dir, 'train')
    val_file_names = get_tfrecord_files(input_dir, 'val')
    checkpoint3 = time.time()

    # next_batch = get_tfrecord_loader(train_file_names, args.batch_size, epochs=args.num_epochs)
    # with tf.Session() as sess:
    #     first_batch = sess.run(next_batch)
    # images = first_batch[0]['input_1']
    # for d in images[:10]:
    #     image = kimg.array_to_img(d)
    #     image.show()
    #     input()
    # print(first_batch[1][:10])

    train_spec = tf.estimator.TrainSpec(lambda: get_tfrecord_loader(train_file_names,
                                                                    args.batch_size,
                                                                    epochs=args.nrof_epochs),
                                        max_steps=args.max_steps)
    eval_spec = tf.estimator.EvalSpec(lambda: get_tfrecord_loader(val_file_names,
                                                                  args.batch_size,
                                                                  shuffle=False))
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    checkpoint4 = time.time()
    next_batch = get_tfrecord_loader(val_file_names, args.batch_size, shuffle=False)
    labels = np.array([])
    with tf.Session() as sess:
        while True:
            try:
                batch = sess.run(next_batch)
            except tf.errors.OutOfRangeError:
                break
            else:
                labels = np.append(labels, batch[1])

    predictions = model.predict(lambda: get_tfrecord_loader(val_file_names, args.batch_size, shuffle=False))
    predictions = np.array(list(map(lambda x: 0 if x['dense_2'][0] < 0.5 else 1, predictions)))
    assert len(predictions) == len(labels)
    print('Accuracy: %.2f%%' % (np.sum(predictions == labels) * 100 / len(labels)))

    checkpoint5 = time.time()

    print('Elapsed time: %.2sf' % (checkpoint5 - start_time))
    print('Create TFRecords: %.2fs' % (checkpoint1 - start_time))
    print('Build model: %.2fs' % (checkpoint2 - checkpoint1))
    print('Get file names: %.2fs' % (checkpoint3 - checkpoint2))
    print('Train and evaluate: %.2fs' % (checkpoint4 - checkpoint3))
    print('Calculate accuracy: %.2fs' % (checkpoint5 - checkpoint4))


if __name__ == '__main__':
    """Parse command line arguments, validate them then invoke main logic"""
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    args.data_dir = config.get(CONFIG_SECTION_INPUT, 'data_dir')
    args.output_dir = config.get(CONFIG_SECTION_OUTPUT, 'output_dir')
    args.log_dir = config.get(CONFIG_SECTION_OUTPUT, 'log_dir')
    args.learning_rate = config.getfloat(CONFIG_SECTION_HYPERPARAMS, 'learning_rate')
    args.momentum = config.getfloat(CONFIG_SECTION_HYPERPARAMS, 'momentum')
    args.nrof_epochs = config.getint(CONFIG_SECTION_HYPERPARAMS, 'nrof_epochs')
    args.max_steps = config.getint(CONFIG_SECTION_HYPERPARAMS, 'max_steps')
    args.batch_size = config.getint(CONFIG_SECTION_HYPERPARAMS, 'batch_size')
    validate_input(args)
    train(args)
