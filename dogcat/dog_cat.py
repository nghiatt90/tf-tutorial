import argparse
import glob
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.data import TFRecordDataset
from tensorflow.python.keras.preprocessing import image as kimg
import time
from typing import Dict, List, Tuple

from preprocess import create_tfrecords


tf.logging.set_verbosity(tf.logging.INFO)

DEFAULT_CONFIG_VALUES = {
    'learning_rate': 1e-3,
    'momentum': 0.9,
    'batch_size': 32,
}
# Default number of classes in pre-trained models (ImageNet)
MODEL_DEFAULT_CLASS_COUNT = 1024
# Number of output classes
OUTPUT_CLASS_COUNT = 2


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
        image = tf.decode_raw(features['image'], tf.float32)
        image = tf.reshape(image, [image_size, image_size, channels])
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


def build_model(lr: float = 1e-3, momentum: float = 0.9, model_dir: str = None) -> tf.estimator.Estimator:
    # Use Keras's Inception v3 model without weights to retrain from scratch.
    # include_top = False removes the last fully connected layer.
    base_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')

    # Replace last layer with our own.
    # GlobalAveragePooling2D converts the MxNxC tensor output into a 1xC tensor where C is the # of channels.
    # Dense is a fully connected layer.
    layers = tf.keras.layers
    model = tf.keras.models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(MODEL_DEFAULT_CLASS_COUNT, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    base_model.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=lr),
        loss='binary_crossentropy',
        metric=['accuracy']
    )
    # print(model.summary())

    model_dir = os.path.join(os.getcwd(), model_dir)
    os.makedirs(model_dir, exist_ok=True)
    return tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir)


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
    model = build_model(args.learning_rate, args.momentum, args.output_dir)
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
                                                                    epochs=args.num_epochs),
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

    print('Elapsed time:', checkpoint5 - start_time)
    print('Create TFRecords:', checkpoint1 - start_time)
    print('Build model:', checkpoint2 - checkpoint1)
    print('Get file names:', checkpoint3 - checkpoint2)
    print('Train and evaluate:', checkpoint4 - checkpoint3)
    print('Calculate accuracy:', checkpoint5 - checkpoint4)


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
    parser.add_argument('--output-dir', '-o', type=str,
                        default='.',
                        help='Path to save trained model. Default: current directory')
    parser.add_argument('--learning-rate', '-lr', type=float,
                        default=DEFAULT_CONFIG_VALUES['learning_rate'],
                        help='Learning rate')
    parser.add_argument('--momentum', '-m', type=float,
                        default=DEFAULT_CONFIG_VALUES['momentum'],
                        help='Momentum')
    parser.add_argument('--num-epochs', '-e', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--max-steps', '-x', type=int,
                        default=1000,
                        help='Maximum number of training steps. Default to 1000')
    parser.add_argument('--batch-size', '-b', type=int,
                        default=DEFAULT_CONFIG_VALUES['batch_size'],
                        help='Number of images to load in every batch')
    parser.add_argument('--test', action='store_true',
                        help='Activate training mode')
    args = parser.parse_args()
    validate_input(args)
    if not args.test:
        train(args)
    else:
        test(args)
