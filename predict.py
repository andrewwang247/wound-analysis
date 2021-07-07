"""Make predictions using encoder."""
# pylint: disable=no-name-in-module
from json import load
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from click import command, option, Path  # type: ignore
from dataset import process_img
from model import HYP_FILE
from center import CENTERS_FILE
from evaluate import load_encoder, get_dense
# pylint: disable=no-value-for-parameter


@command()
@option('--img', '-i', type=Path(exists=True,
                                 file_okay=True,
                                 dir_okay=False),
        required=True, help='Path to input image.')
def predict(img: Image.Image) -> int:
    """Predict a label for the given image."""
    with open(HYP_FILE) as fin:
        hyp: dict = load(fin)
    width = hyp['img_width']
    height = hyp['img_height']
    blur_radius = hyp['blur_radius']
    in_img = process_img(img, width, height, blur_radius)
    encoder = load_encoder(in_img.shape)
    dense = get_dense(in_img.shape)
    centers = np.load(CENTERS_FILE)

    rep = encoder(in_img[np.newaxis, ...])
    outputs = dense(np.abs(centers - rep))
    return np.argmax(outputs)


if __name__ == '__main__':
    predict()
