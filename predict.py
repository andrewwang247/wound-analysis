"""Predict using wound analysis model."""
from typing import List
from json import load
from random import randrange
import numpy as np  # type: ignore
from tensorflow.keras import Model  # type: ignore
from data_process import load_data
from model import get_siamese_model, MODEL_FILE, HYP_FILE


def pick_canonicals(images: np.ndarray, labels: List[int]) -> np.ndarray:
    """Choose a random canonical image from each label."""
    unique_labels = np.unique(labels)
    canon = np.empty((len(unique_labels), *images.shape[1:]))
    splits = [images[labels == unq] for unq in unique_labels]
    choices = [randrange(group.shape[0]) for group in splits]
    for idx, (split, choice) in enumerate(zip(splits, choices)):
        canon[idx, ...] = split[choice]
    return canon


def predict(siamese: Model, images: np.ndarray,
            labels: np.ndarray) -> np.ndarray:
    """Predict using wound analysis model and give 0-1 loss."""
    canon = pick_canonicals(images, labels)
    zero_one = np.empty(len(labels), dtype=bool)
    for idx, (img, lbl) in enumerate(zip(images, labels)):
        img_repeat = np.stack([img] * len(canon))
        output = siamese.predict((canon, img_repeat))
        pred = np.argmax(output.flatten())
        zero_one[idx] = (pred == lbl)
    return zero_one


def main():
    """Run prediction on all images."""
    with open(HYP_FILE) as fin:
        hyp: dict = load(fin)
    images, labels = load_data(hyp)
    siamese = get_siamese_model(images.shape[1:])
    siamese.load_weights(MODEL_FILE)
    print(f'Model weights loaded from {MODEL_FILE}')
    loss = predict(siamese, images, labels)
    print(f'0-1 Accuracy: {np.count_nonzero(loss) / len(loss)}')


if __name__ == '__main__':
    main()
