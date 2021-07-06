"""Load and process images and labels."""
from typing import Tuple
from os import listdir, path
from math import comb
from itertools import combinations
from sklearn.utils import shuffle  # type: ignore
from PIL import Image  # type: ignore
from PIL.ImageFilter import GaussianBlur  # type: ignore
import numpy as np  # type: ignore
from tqdm import tqdm  # type: ignore

TRAIN_DIR = 'images/train'
TEST_DIR = 'images/test'
NUM_LABELS = 5
NORM = (2.0**8 - 1)


def process_img(img: Image.Image,
                width: int,
                height: int,
                blur_radius: int) -> np.ndarray:
    """Serve as placeholder for prediction."""
    pass


def open_img(fpath: str,
             width: int,
             height: int,
             blur_radius: int) -> np.ndarray:
    """Open, resize, and blur image."""
    img = Image.open(fpath) \
        .resize((width, height)) \
        .filter(GaussianBlur(blur_radius))
    return np.asarray_chkfinite(img) / NORM


def load_imgs(
        direc: str,
        width: int,
        height: int,
        blur_radius: int) -> np.ndarray:
    """Load, preprocess, and normalize images from dir."""
    print(f'Loading and processing {direc}...')
    img_lst = listdir(direc)
    images = np.empty((len(img_lst), height, width, 3), dtype=float)
    for idx, fpath in tqdm(enumerate(img_lst), total=len(img_lst)):
        images[idx, ...] = open_img(path.join(direc, fpath),
                                    width, height, blur_radius)
    return images


def load_data(hyp: dict,
              is_test: bool = False) -> Tuple[np.ndarray,
                                              np.ndarray]:
    """Load images and labels with hyperparameters."""
    parent = TEST_DIR if is_test else TRAIN_DIR
    width = hyp['img_width']
    height = hyp['img_height']
    blur = hyp['blur_radius']
    images = [load_imgs(path.join(parent, str(lbl)), width, height, blur)
              for lbl in range(NUM_LABELS)]
    labels = []
    for lbl in range(NUM_LABELS):
        labels += [lbl] * len(listdir(path.join(parent, str(lbl))))
    return np.concatenate(images), np.array(labels)


def generate_pairs(images: np.ndarray, labels: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Generate Siamese image pairs."""
    num_combs = comb(len(labels), 2)
    img_pairs = np.empty((num_combs, 2, *images.shape[1:]))
    print('Generating pairs...')
    for idx, img_pair in tqdm(
        enumerate(
            combinations(
            images, 2)), total=num_combs):
        img_pairs[idx, ...] = np.stack(img_pair)
    lbl_pairs = np.fromiter((left == right for left, right
                             in combinations(labels, 2)), dtype=bool)
    return shuffle(img_pairs, lbl_pairs)


def split_pairs(images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split images into left and right of pairs."""
    return images[:, 0, ...], images[:, 1, ...]
