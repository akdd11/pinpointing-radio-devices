import os

import numpy as np

AP_IDXS = [1, 2, 3]
RF_IDXS = [0, 1]

AGV_SPEED = 0.6


def downsample_subchannels(data, target_num_subchannels):
    """Downsamples the data to the target number of subchannels.

    The downsampling is implemented by adding the powers of the subchannels (in linear scale).

    Input
    -----
    data : np.ndarray
        The data to be downsampled in the shape [num_samples, num_aps, num_rfs, num_subchannels].
    target_num_subchannels : int
        The target number of subchannels.

    Output
    ------
    downsampled_data : np.ndarray
        The downsampled data in the shape [num_samples, num_aps, num_rfs, target_num_subchannels].
    """

    data_lin = 10 ** (data / 10)
    data = np.zeros(
        (data.shape[0], data.shape[1], data.shape[2], target_num_subchannels)
    )

    num_sc_per_group = data_lin.shape[3] // target_num_subchannels

    for new_subcarrier_idx in range(target_num_subchannels):
        data[:, :, :, new_subcarrier_idx] = np.sum(
            data_lin[
                :,
                :,
                :,
                new_subcarrier_idx
                * num_sc_per_group : (new_subcarrier_idx + 1)
                * num_sc_per_group,
            ],
            axis=3,
        )

    data = 10 * np.log10(data)
    return data


def load_one_round(data_path, round_idx, ap_idxs, rf_idxs, num_subchannels):
    """Loads the data for one round.

    Input
    -----
    data_path : str
        The path to the data.
    round_idx : int
        The index of the round.
    ap_idxs : list of int
        The indices of the access points.
    rf_idxs : list of int
        The indices of the RFs.
    num_subchannels : int
        The number of subchannels which shall be loaded.

    Output
    ------
    data : np.ndarray
        Loaded data for the round in the shape [num_samples, num_aps, num_rfs, num_subchannels].
    """

    if num_subchannels != 16 and 40 % num_subchannels != 0:
        raise ValueError("Number of subchannels is not supported")

    if num_subchannels == 1:
        num_subchannels_str = "wideband"
    elif num_subchannels == 16:
        num_subchannels_str = f"16subchannels"
    else:
        num_subchannels_str = "subchannels"

    if num_subchannels not in [1, 16]:
        num_subchannels_loading = 40
    else:
        num_subchannels_loading = num_subchannels

    for ai, ap_idx in enumerate(ap_idxs):
        for ri, rf_idx in enumerate(rf_idxs):
            data_ = np.load(
                os.path.join(
                    data_path,
                    f"iteration_{round_idx}_ap_{ap_idx}_rf_{rf_idx}_{num_subchannels_str}.npy",
                )
            )

            if num_subchannels == 1:
                # make the shape consistent to always have the subchannels as second dimension
                data_ = data_[np.newaxis, :]

            if ai == 0 and ri == 0:
                # in the beginning, allocate the memory
                data = (
                    np.ones(
                        (
                            len(ap_idxs),
                            len(rf_idxs),
                            num_subchannels_loading,
                            data_.shape[1],
                        )
                    )
                    * np.nan
                )

            data[ai, ri, :, :] = (
                data_  # transpose to have the number of samples as the first dimension
            )

    data = np.transpose(data, (3, 0, 1, 2))

    if data.shape[3] != num_subchannels:
        data = downsample_subchannels(data, num_subchannels)

    # cut away samples at the beginning because the AGV does not start to move instantaneously
    # cut away samples at the end because there are NaNs sometimes
    data = data[250:-15, :, :, :]

    return data


def smooth_round(data, sequence_length, num_timesteps=1, stride=1):
    """Smooths the data for one round by for each sample calculating the
    mean over the previous sequence_length samples. If a stride is given,
    the distance between samples considered for averaging is stride."

    Input
    -----
    data : np.ndarray
        The data to be smoothed in the shape [num_samples, num_aps, num_rfs, num_subchannels].
    sequence_length : int
        The length of the sequence to be averaged.
    num_timesteps : int
        The number of steps the sequence is divided into.
        Default: 1
    stride : int
        The step size between the samples considered for averaging.
        Default: 1

    Output
    ------
    smoothed_data : np.ndarray
        The smoothed data in the shape [num_samples, num_aps, num_rfs, num_subchannels, num_timesteps].
    """

    if stride * (num_timesteps - 1) > sequence_length:
        raise ValueError(
            "\nThe sequence length must be long enough that there is at least "
            "one CSI sample per timeblock."
        )

    num_samples, num_aps, num_rfs, num_subchannels = data.shape

    smoothed_data = (
        np.ones(
            (
                num_samples - sequence_length,
                num_aps,
                num_rfs,
                num_subchannels,
                num_timesteps,
            )
        )
        * np.nan
    )

    subsequence_length = sequence_length // num_timesteps

    for ap_idx in range(num_aps):
        for rf_idx in range(num_rfs):
            for subchannel_idx in range(num_subchannels):
                for i in range(sequence_length, num_samples):
                    for ts in range(num_timesteps):
                        start_time_idx = i - sequence_length + subsequence_length * ts
                        end_time_idx = start_time_idx + subsequence_length
                        smoothed_data[
                            i - sequence_length, ap_idx, rf_idx, subchannel_idx, ts
                        ] = np.mean(
                            data[
                                start_time_idx:end_time_idx:stride,
                                ap_idx,
                                rf_idx,
                                subchannel_idx,
                            ]
                        )

    return smoothed_data


def get_results_filename(
    results_path, method, seq_len, interval, num_subchannels, num_timesteps, **kwargs
):
    """Returns the filename in which the results are saved for the given configuration.

    Input
    -----
    results_path : str
        The path to the results folder.
    method : str
        The method used for localization.
    seq_len : int
        The length of the sequence used for smoothing.
    interval : int
        The interval in which the channel is sensed.
    num_subchannels : int
        The number of subchannels used.
    num_timesteps : int
        The number of timesteps in which the sequence is split.
    (optional) num_neighbors : int
        The number of neighbors used for KNN localization.

    Output
    ------
    filename : str
        The filename in which the results are saved.
    """
    if method == "knn":
        num_neighbors = kwargs["num_neighbors"]
        filename = (
            f"{method}{num_neighbors}-loc_errors_seq{seq_len}_i{interval}"
            f"_s{num_subchannels}_t{num_timesteps}.npy"
        )
    else:
        filename = (
            f"{method}-loc_errors_seq{seq_len}_i{interval}"
            f"_s{num_subchannels}_t{num_timesteps}.npy"
        )

    return os.path.join(results_path, filename)
