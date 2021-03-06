"""Train wound analysis model."""
# pylint: disable=no-name-in-module
from json import load
from os import environ
from typing import Dict
import numpy as np  # type: ignore
from tensorflow.keras import Model, Sequential  # type: ignore
from tensorflow.keras.layers import InputLayer  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
from dataset import generate_pairs, load_data, split_pairs
from model import get_siamese_model
from model import SIAMESE_FILE, HYP_FILE, ENCODER_FILE, DENSE_FILE


def gpu_init():
    """Set CUDA GPU environment."""
    environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    environ['CUDA_VISIBLE_DEVICES'] = '0'


def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """Compute class weights from labels."""
    num_true = np.count_nonzero(y_train)
    num_false = len(y_train) - num_true
    weights = len(y_train) / np.array([num_true, num_false]) / 2
    return dict(zip([True, False], weights / np.sum(weights)))


def extract_encoder(siamese: Model) -> Model:
    """Extract encoder part of siamese network."""
    enc_layer = siamese.get_layer('encoder')
    return Model(inputs=enc_layer.input,
                 outputs=enc_layer.output)


def extract_dense(siamese: Model) -> Model:
    """Extract dense part of siamese network."""
    return Sequential([
        InputLayer((None, 2048)),
        siamese.layers[-1]
    ], name='dense')


def train():
    """Train wound analysis model."""
    with open(HYP_FILE) as fin:
        hyp: dict = load(fin)
    images, labels = load_data(hyp)
    img_pairs, lbl_pairs = generate_pairs(images, labels)
    siamese = get_siamese_model(images.shape[1:])
    siamese.summary()
    siamese.compile(optimizer=Adam(learning_rate=hyp['learning_rate']),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    try:
        siamese.fit(split_pairs(img_pairs), lbl_pairs,
                    epochs=hyp['epochs'],
                    batch_size=hyp['batch_size'],
                    class_weight=compute_class_weights(lbl_pairs),
                    validation_split=hyp['val_size'],
                    callbacks=[ModelCheckpoint(SIAMESE_FILE,
                                               save_best_only=True)])
    finally:
        siamese.load_weights(SIAMESE_FILE)
        print('Siamese weights saved to', SIAMESE_FILE)
        encoder = extract_encoder(siamese)
        encoder.save_weights(ENCODER_FILE)
        print('Encoder weights saved to', ENCODER_FILE)
        dense = extract_dense(siamese)
        dense.save_weights(DENSE_FILE)
        print('Dense weights saved to', DENSE_FILE)


if __name__ == '__main__':
    gpu_init()
    train()
