import argparse
import os
import tensorflow as tf
from typing import List


DEFAULT_CONFIG_VALUES = {
    'learning_rate': 1e-3,
}
# Default number of classes in pre-trained models (ImageNet)
MODEL_DEFAULT_CLASS_COUNT = 1024


def validate_input(user_inputs: argparse.Namespace) -> None:
    """Validate user input"""

    # args.data must be a valid directory
    data_path = user_inputs.data
    assert os.path.exists(data_path), 'Cannot find data directory: %s' % data_path
    assert os.path.isdir(data_path), '%s is not a directory' % data_path

    # Check directory structure
    # root/
    # root/train_dir
    # root/train_dir/cat
    # root/train_dir/dog
    # root/val_dir
    # root/val_dir/cat
    # root/val_dir/dog
    def get_sub_dirs(path: str) -> List[str]:
        return [sub_dir for sub_dir in os.listdir(path) if os.path.isdir(os.path.join(path, sub_dir))]

    sub_dirs = get_sub_dirs(data_path)
    assert 'train_dir' in sub_dirs, 'Cannot find "train_dir" sub-directory under %s' % data_path
    assert 'val_dir' in sub_dirs, 'Cannot find "val_dir" sub-directory under %s' % data_path

    train_path = os.path.join(data_path, 'train_dir')
    train_sub_dirs = get_sub_dirs(train_path)
    assert 'dog' in train_sub_dirs, 'Cannot find "dog" sub-directory under %s' % train_path
    assert 'cat' in train_sub_dirs, 'Cannot find "cat" sub-directory under %s' % train_path

    val_path = os.path.join(data_path, 'val_dir')
    val_sub_dirs = get_sub_dirs(val_path)
    assert 'dog' in val_sub_dirs, 'Cannot find "dog" sub-directory under %s' % val_path
    assert 'cat' in val_sub_dirs, 'Cannot find "cat" sub-directory under %s' % val_path


def get_data_loader(data_path: str):
    pass


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


if __name__ == '__main__':
    """Parse command line arguments, validate them then invoke main logic"""
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str,
                        help='Path to data directory')
    parser.add_argument('--learning-rate', '-lr', type=float,
                        default=DEFAULT_CONFIG_VALUES['learning_rate'],
                        help='Learning rate')
    parser.add_argument('--momentum', '-m', type=float,
                        help='Momentum')
    parser.add_argument('--train', action='store_true',
                        help='Activate training mode')
    args = parser.parse_args()
    validate_input(args)


