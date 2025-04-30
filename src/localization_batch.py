import pandas as pd
from tqdm import trange

import data_utils
from localization import localization

TEST_STRIDE = 50

DATA_PATH = r"data"

AP_IDXS = data_utils.AP_IDXS
RF_IDXS = data_utils.RF_IDXS


batch_parameters = pd.read_excel("results_to_generate.xlsx")

# in the style the excel file is created, there are some duplicates
batch_parameters = batch_parameters.drop_duplicates(ignore_index=True)

for idx in trange(len(batch_parameters)):

    method = batch_parameters["Method"][idx]
    sequence_length = batch_parameters["Sequence Length"][idx].astype(int)
    num_subchannels = batch_parameters["Num Subchannels"][idx].astype(int)
    num_timesteps = batch_parameters["Num Timesteps"][idx].astype(int)
    interval = batch_parameters["Interval"][idx].astype(int)
    num_neighbors = batch_parameters["Num Neighbors"][idx].astype(int)
    save_distances = batch_parameters["Save Distances"][idx]

    try:
        localization(
            DATA_PATH,
            AP_IDXS,
            RF_IDXS,
            method,
            sequence_length,
            num_subchannels,
            num_timesteps,
            interval,
            TEST_STRIDE,
            num_neighbors=num_neighbors,
            save_neighbor_distance=save_distances,
        )
    except Exception as e:
        print(
            f"Error in round {idx}: {e} Parameters: {method}, {sequence_length},"
            f"{num_subchannels}, {num_timesteps}, {interval}, {num_neighbors}"
        )
