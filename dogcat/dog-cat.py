import argparse
import glob
import os
import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset
from typing import Any, Dict, List, Tuple

from preprocess import create_tfrecords


DEFAULT_CONFIG_VALUES = {
    'learning_rate': 1e-3,
    'batch_size': 128,
}
# Default number of classes in pre-trained models (ImageNet)
MODEL_DEFAULT_CLASS_COUNT = 1024


def validate_input(user_inputs: argparse.Namespace) -> None:
    """Validate user input"""

    # args.data must be a valid directory
    data_path = user_inputs.data_dir
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


def _parse_example_proto(proto: tf.train.Example,
                         image_size: int = 299, channels: int = 3)\
        -> Tuple[tf.Tensor, tf.Tensor]:
    """

    :param proto:
    :return:
    """
    feature_map = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }
    features = tf.parse_example(proto, features=feature_map)
    images = tf.decode_raw(features['image'], tf.float32)
    images = tf.reshape(images, [-1, image_size, image_size, channels])
    labels = tf.cast(features['label'], tf.float32)
    return images, labels


def get_tfrecord_loader(file_names: tf.placeholder, batch_size: int = None, buffer_size: int = 1000)\
        -> tf.contrib.data.Iterator:
    """Create a dataset to read tfrecord files and return its iterator.

    The iterator expects a list of tfrecord file names to be fed to
    its 'file_names' placeholder.

    :param file_names: tf.placeholder
    :param batch_size: Number of images in each batch
    :param buffer_size: See tf.contrib.Dataset.shuffle
    :return: Tensor of type Iterator
    """
    dataset = TFRecordDataset(file_names)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_parse_example_proto)
    return dataset.make_initializable_iterator()


def build_model(n_classes: int):
    # Use Keras's Inception v3 model without weights to retrain from scratch.
    # include_top = False removes the last fully connected layer.
    model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights=None)

    # Replace last layer with our own.
    # GlobalAveragePooling2D converts the MxNxC tensor output into a 1xC tensor where C is the # of channels.
    # Dense is a fully connected layer.
    last_layer = model.output
    last_layer = tf.keras.layers.GlobalAveragePooling2D()(last_layer)
    last_layer = tf.keras.layers.Dense(MODEL_DEFAULT_CLASS_COUNT, activation='relu')(last_layer)
    predictions = tf.keras.layers.Dense(n_classes, activation='softmax')(last_layer)  # new softmax layer
    model = tf.keras.models.Model(inputs=model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=args.learning_rate, momentum=args.momentum),
        loss='categorical_crossentropy',
        metric='accuracy'
    )
    return tf.keras.estimator.model_to_estimator(keras_model=model)


# noinspection PyShadowingNames
def train(args: argparse.Namespace) -> None:
    """
    Train a new model.

    :param args: Validated user inputs
    :return: None
    """
    input_dir = create_tfrecords(args.data_dir)
    file_names = tf.placeholder(tf.string, shape=[None])
    iterator = get_tfrecord_loader(file_names, args.batch_size)
    next_batch = iterator.get_next()

    train_file_names = get_tfrecord_files(input_dir, 'train')
    with tf.Session() as session:
        for epoch_id in range(args.num_epochs):
            session.run(iterator.initializer, feed_dict={file_names: train_file_names})
            while True:
                try:
                    images, labels = session.run(next_batch)

                except tf.errors.OutOfRangeError:
                    # End of 1 epoch
                    break


# noinspection PyShadowingNames
def test(args: argparse.Namespace) -> None:
    """
    Test a trained model.

    :param args: Validated user inputs
    :return: None
    """
    pass


if __name__ == '__main__':
    """Parse command line arguments, validate them then invoke main logic"""
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help='Path to extracted training data')
    parser.add_argument('--learning-rate', '-lr', type=float,
                        default=DEFAULT_CONFIG_VALUES['learning_rate'],
                        help='Learning rate')
    parser.add_argument('--momentum', '-m', type=float,
                        help='Momentum')
    parser.add_argument('--num-epochs', '-e', type=int, default=1,
                        help='Number of training epochs. Ignored if --train is not specified')
    parser.add_argument('--batch-size', '-b', type=int,
                        default=DEFAULT_CONFIG_VALUES['batch_size'],
                        help='Number of images to load in every batch')
    parser.add_argument('--train', action='store_true',
                        help='Activate training mode')
    args = parser.parse_args()
    validate_input(args)
    if args.train:
        train(args=args)
