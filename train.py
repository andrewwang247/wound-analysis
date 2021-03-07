"""
Train wound analysis model.

Copyright 2021. Siwei Wang.
"""
from json import load
from sklearn.model_selection import train_test_split  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.metrics import BinaryAccuracy  # type: ignore
from data_process import gpu_init, generate_pairs, load_data, split_pairs
from model import get_siamese_model, MODEL_FILE, HYP_FILE


def train():
    """Train wound analysis model."""
    with open(HYP_FILE) as fin:
        hyp: dict = load(fin)
    images, labels = load_data(hyp)
    img_pairs, lbl_pairs = generate_pairs(images, labels)
    x_train, x_test, y_train, y_test = train_test_split(
        img_pairs, lbl_pairs, test_size=hyp['test_size'], stratify=lbl_pairs)
    siamese = get_siamese_model(images.shape[1:])
    siamese.summary()
    siamese.compile(optimizer=Adam(lr=hyp['learning_rate']),
                    loss='binary_crossentropy',
                    metrics=[BinaryAccuracy()])
    siamese.fit(split_pairs(x_train), y_train,
                epochs=hyp['epochs'], batch_size=hyp['batch_size'])
    siamese.evaluate(split_pairs(x_test), y_test)
    siamese.save_weights(MODEL_FILE)
    print(f'Model weights saved to {MODEL_FILE}')


if __name__ == '__main__':
    gpu_init()
    train()
