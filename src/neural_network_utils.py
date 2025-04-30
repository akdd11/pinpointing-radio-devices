import numpy as np

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Add
from tensorflow.keras.models import Model


def build_mlp(input_shape, verbose=0):
    # number of nodes so that the number of trainable parameters
    # scales very roughly linearly with the input shape
    node_scaler = 1 if input_shape[0] / 6 == 1 else np.log2(input_shape[0] / 6)
    nodes_per_layer = int(128 * node_scaler)

    inputs = Input(shape=input_shape)
    x = Dense(nodes_per_layer, activation="elu")(inputs)
    x = Dense(nodes_per_layer, activation="elu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)

    model.compile(optimizer="adam", loss="mean_squared_error")

    if verbose:
        model.summary()

    return model


def residual_block(x, units, dropout_rate=0.2):
    """Defines a residual block with skip connection"""
    shortcut = x  # Save input for the skip connection

    # Fully Connected Layer
    x = Dense(units, activation="relu")(x)
    x = BatchNormalization()(x)  # Normalize for stability
    x = Dropout(dropout_rate)(x)  # Regularization

    # Fully Connected Layer
    x = Dense(units, activation="relu")(x)
    x = BatchNormalization()(x)

    # Skip Connection (adds input to output)
    x = Add()([x, shortcut])
    return x


def build_residual_mlp(
    input_dim,
    hidden_units=[64, 64],
    dropout_rate=0.2,
    verbose=0,
):
    """Builds a Residual MLP model"""
    inputs = Input(shape=(input_dim,))
    x = Dense(hidden_units[0], activation="relu")(inputs)  # Initial Dense Layer
    x = BatchNormalization()(x)

    # Apply Residual Blocks
    for units in hidden_units:
        x = residual_block(x, units, dropout_rate)

    # Output Layer
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    if verbose:
        model.summary()

    return model
