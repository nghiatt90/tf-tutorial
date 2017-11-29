import argparse
import os
from typing import List


DEFAULT_CONFIG_VALUES = {
    'learning_rate': 1e-3,
}


def validate_input(args: argparse.Namespace) -> None:
    """Validate user input"""

    # namespace.data must be a valid directory
    data_path = args.data
    assert os.path.exists(data_path), 'Cannot find data directory: %s' % data_path
    assert os.path.isdir(data_path), '%s is not a directory' % data_path

    # Check directory structure
    def get_sub_dirs(path: str) -> List[str]:
        return [sub_dir for sub_dir in os.listdir(path) if os.path.isdir(os.path.join(path, sub_dir))]

    sub_dirs = get_sub_dirs(data_path)
    assert 'train' in sub_dirs, 'Cannot find "train" sub-directory under %s' % data_path
    assert 'val' in sub_dirs, 'Cannot find "val" sub-directory under %s' % data_path

    train_path = os.path.join(data_path, 'train')
    train_sub_dirs = get_sub_dirs(train_path)
    assert 'dog' in train_sub_dirs, 'Cannot find "dog" sub-directory under %s' % train_path
    assert 'cat' in train_sub_dirs, 'Cannot find "cat" sub-directory under %s' % train_path

    val_path = os.path.join(data_path, 'val')
    val_sub_dirs = get_sub_dirs(val_path)
    assert 'dog' in val_sub_dirs, 'Cannot find "dog" sub-directory under %s' % val_path
    assert 'cat' in val_sub_dirs, 'Cannot find "cat" sub-directory under %s' % val_path


if __name__ == '__main__':
    """Parse command line arguments, validate them then invoke main logic"""

    # Declare and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str,
                        help='Path to data directory')
    parser.add_argument('--learning-rate', '-lr', type=float,
                        default=DEFAULT_CONFIG_VALUES['learning_rate'],
                        help='Learning rate')
    parser.add_argument('--train', action='store_true',
                        help='Activate training mode')
    namespace = parser.parse_args()
    validate_input(namespace)
