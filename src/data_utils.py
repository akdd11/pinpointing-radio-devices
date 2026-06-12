import os
import zipfile

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

def cfr_to_subband_channel_gain(arr_original, n_subbands):
    """Convert complex CFR data to logarithmic subband channel gain.

    Parameters
    ----------
    arr_original : np.ndarray
        Input array with shape (n_samples, n_freq_bins).
    n_subbands : int
        Number of subbands to create, e.g. 40 or 16.

    Returns
    -------
    np.ndarray
        Array with shape (n_samples, n_subbands) containing 10*log10 of the
        summed subband energy.
    """
    arr_original = np.asarray(arr_original)
    if arr_original.ndim != 2:
        raise ValueError(f"arr_original must be 2D, got shape {arr_original.shape}")

    energy = np.abs(arr_original) ** 2
    subband_energy = np.stack(
        [energy[:, idx].sum(axis=1) for idx in np.array_split(np.arange(energy.shape[1]), n_subbands)],
        axis=1,
    )

    return 10 * np.log10(np.maximum(subband_energy, np.finfo(float).tiny))


def load_one_round(data_path, round_idx, ap_idxs, rf_idxs, num_subchannels):
    """Loads the data for one round from the zipped files (IEEE DataPort).

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

    for ap_idx in ap_idxs:
        for rf_idx in rf_idxs:

            filename = os.path.join(data_path, f"scenario_index_{round_idx}.zip")
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                with zip_ref.open(f"cirs_scenario_{round_idx}_rx_{ap_idx}_rf_{rf_idx}.npy") as f:
                    # the original file contains the CIRs in shape [num_samples, num_taps]
                    cirs = np.load(f)

            cirs = cirs.T
            cfrs = np.fft.fftshift(np.fft.fft(cirs, axis=1), axes=1)

            data_ = cfr_to_subband_channel_gain(cfrs, num_subchannels)
            if data_.shape[0] != 33000:
                data_ = data_[:33000, :]

            if ap_idx == ap_idxs[0] and rf_idx == rf_idxs[0]:
                # in the beginning, allocate the memory
                data = (
                    np.ones(
                        (
                            data_.shape[0],
                            len(ap_idxs),
                            len(rf_idxs),
                            num_subchannels,
                        )
                    )
                    * np.nan
                )

            data[:, ap_idx-1, rf_idx, :] = data_

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
