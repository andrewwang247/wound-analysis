"""Load and process images and labels."""
from typing import Tuple
from os import environ, listdir, path
from scipy.special import comb
from itertools import combinations
from PIL import Image  # type: ignore
from PIL.ImageFilter import GaussianBlur  # type: ignore
import numpy as np  # type: ignore

IMG_DIR = 'images'
LABEL_FILE = 'labels.txt'
NORM = (2.0**8 - 1)


def gpu_init():
    """Set CUDA GPU environment."""
    environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    environ['CUDA_VISIBLE_DEVICES'] = '0'


def open_img(fpath: str,
             width: int,
             height: int,
             blur_radius: int,
             add_path=True) -> np.ndarray:
    """Open, resize, and blur image."""
    img = Image.open(path.join(IMG_DIR, fpath)
                     if add_path else fpath)
    return process_img(img, width, height, blur_radius)


def process_img(img: Image.Image,
                width: int,
                height: int,
                blur_radius: int) -> np.ndarray:
    """Preprocess image according to spec."""
    proc = img.resize((width, height)).filter(GaussianBlur(blur_radius))
    return np.asarray_chkfinite(proc) / NORM


def load_imgs(width: int, height: int, blur_radius: int) -> np.ndarray:
    """Load, preprocess, and normalize images from IMG_DIR."""
    direc = listdir(IMG_DIR)
    images = np.empty((len(direc), height, width, 3), dtype=float)
    for idx, fpath in enumerate(direc):
        images[idx, ...] = open_img(fpath, width, height, blur_radius)
    return images


def load_data(hyp: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Load images and labels with hyperparameters."""
    print('Loading and processing data...')
    images = load_imgs(
        hyp['img_width'],
        hyp['img_height'],
        hyp['blur_radius'])
    labels = np.loadtxt(LABEL_FILE, dtype=int)
    return images, labels


def generate_pairs(images: np.ndarray, labels: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Generate Siamese image pairs."""
    num_combs = comb(len(labels), 2)
    img_pairs = np.empty((num_combs, 2, *images.shape[1:]))
    for idx, img_pair in enumerate(combinations(images, 2)):
        img_pairs[idx, ...] = np.stack(img_pair)
    lbl_pairs = np.fromiter((left == right for left, right
                             in combinations(labels, 2)), dtype=bool)
    return img_pairs, lbl_pairs


def split_pairs(images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split images into left and right of pairs."""
    return images[:, 0, ...], images[:, 1, ...]
