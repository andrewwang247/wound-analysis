"""Define Siamese network model."""
# pylint: disable=no-name-in-module
from typing import Tuple
from tensorflow.keras import Input, Model, Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D  # type: ignore
from tensorflow.keras.layers import Flatten, Dense, Lambda  # type: ignore
from tensorflow.keras.regularizers import l2  # type: ignore
from tensorflow.keras.backend import abs as K_abs  # type: ignore


SIAMESE_FILE = 'siamese.h5'
ENCODER_FILE = 'encoder.h5'
DENSE_FILE = 'dense.h5'
HYP_FILE = 'hyperparameters.json'


def get_encoder_model(input_shape: Tuple[int, ...]) -> Model:
    """Get encoder submodel given input shape."""
    model = Sequential(name='encoder')
    model.add(Conv2D(64, (9, 9), input_shape=input_shape,
                     activation='relu',
                     padding='same',
                     kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu',
                     padding='same',
                     kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (5, 5), activation='relu',
                     padding='same',
                     kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (3, 3), activation='relu',
                     padding='same',
                     kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(2048, activation='sigmoid',
                    kernel_regularizer=l2(1e-3)))
    return model


def get_siamese_model(input_shape: Tuple[int, ...]) -> Model:
    """Define and return siamese model given input shape."""
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = get_encoder_model(input_shape)

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    l1_layer = Lambda(lambda tensors: K_abs(tensors[0] - tensors[1]))
    l1_distance = l1_layer([encoded_l, encoded_r])
    pred = Dense(1, activation='sigmoid')(l1_distance)

    return Model(inputs=[left_input, right_input],
                 outputs=pred,
                 name='siamese')
