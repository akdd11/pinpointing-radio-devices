import os

from sklearnex import patch_sklearn

patch_sklearn()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import data_utils as du
import neural_network_utils as nnu

from data_utils import AGV_SPEED


def localization(
    data_path,
    ap_idxs,
    rf_idxs,
    method,
    sequence_length,
    num_subchannels,
    num_timesteps,
    interval,
    test_stride,
    verbose=0,
    show_nn_train_history=False,
    save_neighbor_distance=False,
    **kwargs,
):
    """Executes the selected localization method with given configuration for
    all test rounds and saves the resulting errors as numpy file.

    Input
    -----
    data_path: str
        Path to the data folder.
    ap_idxs: list
        List of access point indices.
    rf_idxs: list
        List of antenna indices.
    method: str
        Method to be used for localization. Options are "knn", "mlp" or "resmlp".
    sequence_length: int
        Length of the sequence in samples. Note, that the used number of samples
        is smaller than sequence length, if interval > 1.
    num_subchannels: int
        Number of subchannels used for localization.
    num_timesteps: int
        Number of timesteps in which the sequence is split.
    interval: int
        How much time is between two used CSI measurements in samples.
    test_stride: int
        Distance between test points in samples.
    verbose: int
        Verbosity level. 0 = no output, >0 = output.
        Mostly used for MLP and ResMLP.
    show_nn_train_history: bool
        If True, the training history of the neural network is plotted.
        Default: False.
    save_neighbor_distance: bool
        If True, the distance to the nearest neighbor is saved.
        Default: False.
    num_neighbors: int
        Number of neighbors to be used for KNN method.
        Keyword argument, needs to be explicitly specified when KNN is used.
    """

    round_idxs = pd.read_csv("round_idxs.csv")["round_idxs"].values

    reference_idx = round_idxs[0]
    test_idxs = round_idxs[1:]

    reference_data = du.load_one_round(
        data_path, reference_idx, ap_idxs, rf_idxs, num_subchannels
    )

    loc = np.arange(sequence_length, reference_data.shape[0]) * AGV_SPEED

    reference_data_smoothed = du.smooth_round(
        reference_data, sequence_length, num_timesteps=num_timesteps, stride=interval
    )

    if method == "knn":
        try:
            num_neighbors = kwargs["num_neighbors"]
        except KeyError:
            raise ValueError("num_neighbors not specified for KNN method.")

        knn_reg = KNeighborsRegressor(
            n_neighbors=num_neighbors, weights="distance", n_jobs=8
        )
        knn_reg = knn_reg.fit(
            reference_data_smoothed.reshape([reference_data_smoothed.shape[0], -1]), loc
        )

    elif "mlp" in method:

        input_scaler = StandardScaler()
        output_scaler = MinMaxScaler()

        reference_data_smoothed_scaled = input_scaler.fit_transform(
            reference_data_smoothed.reshape([reference_data_smoothed.shape[0], -1])
        )
        loc_scaled = output_scaler.fit_transform(loc.reshape(-1, 1))

        if method == "mlp":
            model = nnu.build_mlp(
                input_shape=[reference_data_smoothed_scaled.shape[1]], verbose=verbose
            )
        elif method == "resmlp":
            nodes_scaler = (
                1
                if num_subchannels * num_timesteps == 1
                else np.log2(num_subchannels * num_timesteps)
            )
            nodes_per_layer = int(64 * nodes_scaler)

            model = nnu.build_residual_mlp(
                input_dim=reference_data_smoothed_scaled.shape[1],
                hidden_units=[nodes_per_layer, nodes_per_layer],
                dropout_rate=0.2,
            )
        else:
            raise ValueError("Unknown method")

        # manual shuffling so that the validation data is not only from the end
        shuffle_idxs = np.random.permutation(len(reference_data_smoothed_scaled))
        reference_data_smoothed_scaled = reference_data_smoothed_scaled[shuffle_idxs]
        loc_scaled = loc_scaled[shuffle_idxs]

        history = model.fit(
            reference_data_smoothed_scaled,
            loc_scaled,
            epochs=50,
            verbose=verbose,
            validation_split=0.02,
        )

        if show_nn_train_history:
            plt.plot(history.history["loss"], label="loss")
            plt.plot(history.history["val_loss"], label="val_loss")
            plt.legend()
            plt.show()
    else:
        raise ValueError("Unknown method")

    # Testing
    loc_test = loc[::test_stride]
    errors = np.ones((len(test_idxs), len(loc_test)))

    if save_neighbor_distance:
        neighbor_distances = np.zeros((len(test_idxs), len(loc_test)))

    for round_idx_cont, round_idx in enumerate(tqdm(test_idxs, disable=(verbose == 0))):
        test_data = du.load_one_round(
            data_path, round_idx, ap_idxs, rf_idxs, num_subchannels
        )
        test_data = du.smooth_round(
            test_data, sequence_length, num_timesteps=num_timesteps, stride=interval
        )

        # reduce the number of test points
        test_data = test_data[::test_stride, :, :, :, :]

        if "mlp" in method:
            test_data_scaled = input_scaler.transform(
                test_data.reshape([test_data.shape[0], -1])
            )
            loc_hat_scaled = model.predict(test_data_scaled, verbose=verbose)
            loc_hat = output_scaler.inverse_transform(loc_hat_scaled).flatten()
        elif method == "knn":
            loc_hat = knn_reg.predict(test_data.reshape([test_data.shape[0], -1]))
            if save_neighbor_distance:
                neighbor_distances[round_idx_cont, :] = knn_reg.kneighbors(
                    test_data.reshape([test_data.shape[0], -1]), return_distance=True
                )[0][:, 0]

        errors[round_idx_cont, :] = loc_hat - loc_test

    results_filename = du.get_results_filename(
        "results",
        method,
        sequence_length,
        interval,
        num_subchannels,
        num_timesteps,
        num_neighbors=num_neighbors if method == "knn" else None,
    )

    np.save(results_filename, errors)
    if save_neighbor_distance:
        np.save(
            os.path.join(
                os.path.dirname(results_filename),
                "neighbor_distances",
                os.path.basename(results_filename),
            ),
            neighbor_distances,
        )


if __name__ == "__main__":

    DATA_PATH = r"data"

    METHOD = "knn"  # knn, mlp or resmlp

    N_SUBCHANNELS = 1  # number of subchannels used for localization
    N_TIMESTEPS = 1  # number of timesteps in which the sequence is split
    SEQUENCE_LENGTH = 250  # in samples
    INTERVAL = 1  # how often the RSS is measured, in samples (milliseconds)

    if METHOD == "knn":
        N_NEIGHBORS = 1

    TEST_STRIDE = 50  # distance between test points in samples

    AP_IDXS = du.AP_IDXS
    RF_IDXS = du.RF_IDXS

    localization(
        DATA_PATH,
        AP_IDXS,
        RF_IDXS,
        METHOD,
        SEQUENCE_LENGTH,
        N_SUBCHANNELS,
        N_TIMESTEPS,
        INTERVAL,
        TEST_STRIDE,
        num_neighbors=N_NEIGHBORS if METHOD == "knn" else None,
        verbose=1,
    )
