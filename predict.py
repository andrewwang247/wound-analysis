"""Make predictions using encoder."""
from json import load
from click import command, option, Path
from data_process import open_img
from model import get_encoder_model, ENCODER_FILE, HYP_FILE


def load_encoder(width: int, height: int):
    """Load encoder with weights."""
    shape = (width, height, 3)
    encoder = get_encoder_model(shape)
    encoder.load_weights(ENCODER_FILE)
    return encoder


@command()
@option('--image', '-i', type=Path(exists=True), required=True,
        help='Path to prediction image.')
def predict(image: str):
    """Make predictions using encoder."""
    with open(HYP_FILE) as fin:
        hyp: dict = load(fin)
    width = hyp['img_width']
    height = hyp['img_height']
    encoder = load_encoder(width, height)
    blur = hyp['blur_radius']
    img = open_img(image, width, height, blur)
    print(img.shape)


if __name__ == '__main__':
    predict()
