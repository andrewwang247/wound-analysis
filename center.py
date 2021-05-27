"""Compute cluster centers based on encoding."""
from json import load
import numpy as np  # type: ignore
from data_process import load_data
from model import get_encoder_model, ENCODER_FILE, HYP_FILE


CENTERS_FILE = 'centers.npy'


def center_clusters(encodings: np.ndarray,
                    labels: np.ndarray) -> np.ndarray:
    """Compute centers of all label clusters."""
    num_unq = len(np.unique(labels))
    centers = np.empty((num_unq, encodings.shape[-1]))
    for unq in range(num_unq):
        centers[unq, ...] = np.mean(encodings[labels == unq], axis=0)
    return centers


def main():
    """Run prediction on all images."""
    with open(HYP_FILE) as fin:
        hyp: dict = load(fin)
    images, labels = load_data(hyp)
    encoder = get_encoder_model(images.shape[1:])
    encoder.load_weights(ENCODER_FILE)
    centers = center_clusters(encoder(images), labels)
    print('Finished computing cluster centers.')
    np.save(CENTERS_FILE, centers)


if __name__ == '__main__':
    main()
