import argparse
import glob
import os
from typing import Dict, List, Tuple

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf

from dog_cat_train import build_model
import estimator_util as eutil


tf.logging.set_verbosity(tf.logging.FATAL)


# noinspection PyShadowingNames
def validate_input(args: argparse.Namespace) -> None:
    """Validate user input"""

    # args.model must be a valid checkpoint
    assert os.path.exists(args.model), 'Cannot find %s' % args.model
    # assert os.path.isfile(args.model), 'Model must be a valid checkpoint file'
    # assert args.model.endswith('.ckpt'), 'Model must be a valid checkpoint file'

    # args.data must exist
    assert os.path.exists(args.data), 'Cannot find %s' % args.data

    if os.path.isfile(args.data):
        assert args.data.endswith(('jpg', 'jpeg', 'JPG', 'JPEG')), 'Data must be an image or a directory'


# noinspection PyShadowingNames
def test(args: argparse.Namespace) -> None:

    image_paths = []
    if os.path.isfile(args.data):
        image_paths = [args.data]
    else:
        image_paths.extend(glob.glob(os.path.join(args.data, '*.jpg')))
        image_paths.extend(glob.glob(os.path.join(args.data, '*.jpeg')))
        image_paths.extend(glob.glob(os.path.join(args.data, '*.JPG')))
        image_paths.extend(glob.glob(os.path.join(args.data, '*.JPEG')))
        assert len(image_paths) > 0, '%s contains no suitable images' % args.data

    images = list(map(lambda x: eutil.load_image(x), image_paths))

    # noinspection PyShadowingNames
    def input_fn(images: List[str]) -> Tuple[List[Dict[str, tf.Tensor]], List[tf.Tensor]]:
        features = []
        for image in images:
            image_tensor = tf.stack([image])
            features.append({'input_1': image_tensor})
        return features

    model_dir = os.path.dirname(args.model)
    model = build_model(model_dir=model_dir)
    with tf.device('/cpu:0'):
        predictions = list(model.predict(lambda: input_fn(images)))
    # predictions = list(map(lambda x: 0 if x['dense_2'][0] < 0.5 else 1, predictions))
    print(predictions)


if __name__ == '__main__':
    """Parse command line arguments, validate them then invoke main logic"""
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='Path to save trained model.')
    parser.add_argument('data', type=str,
                        help='Path to testing data')
    args = parser.parse_args()
    validate_input(args)
    test(args)
