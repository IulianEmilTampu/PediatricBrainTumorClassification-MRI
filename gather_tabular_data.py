# %%
"""
Script that gathers the tabular data from the repeated cross-validation trainings.
It returns a .csv file ready for plotting using the implemented plotting utilities.

Steps
1 - gets the necessary paths of where the models are located
2 - for every model, opens the best and last tabular .csv files and stores the
    performance values.
3 - reorganize all the stored values and save summary csv file.
"""
import os
import sys
import csv
import glob
import argparse
import pandas as pd
from pathlib import Path

# %%
su_debug_flag = True

# --------------------------------------
# read the input arguments and set the base folder
# --------------------------------------
if not su_debug_flag:
    parser = argparse.ArgumentParser(
        description="Script that gathers the logits data from the repeated cross-validation trainings."
    )
    parser.add_argument(
        "-ptm",
        "--PATH_TO_MODELS",
        required=True,
        help="Path to where the folder containing the repeated cross validation models are saved.",
    )

    args_dict = dict(vars(parser.parse_args()))
else:
    print("Running in debug mode.")
    args_dict = {
        "PATH_TO_MODELS": "/flush/iulta54/Research/P5-MICCAI2023/trained_models_archive/Classification_optm_ADAM_SDM4_TFRdata_True_modality_T2_loss_MCC_and_CCE_Loss_lr_0.0001_batchSize_32_pretrained_False_frozenWeight_True_useAge_True_large_age_encoder_useGradCAM_False",
    }

args_dict["SAVE_PATH"] = os.path.join(
    args_dict["PATH_TO_MODELS"], "summary_aggregation"
)
Path(args_dict["SAVE_PATH"]).mkdir(parents=True, exist_ok=True)

# print input variables
max_len = max([len(key) for key in args_dict])
[print(f"{key:{max_len}s}: {value}") for key, value in args_dict.items()]

# %% OPEN SUMMARY FILES FOR EVERY MODEL AND STORE DATA
# get list of repeated cross validation folders
CV_REPETITION_FOLDERS = glob.glob(
    os.path.join(args_dict["PATH_TO_MODELS"], "*_seed_*", "")
)

# get path to the tabular_test_summary.csv file for all the models
CV_TEST_SUMMARY_FILES = [
    os.path.join(mp, "test_summary", "tabular_test_summary.csv")
    for mp in CV_REPETITION_FOLDERS
    if os.path.isfile(os.path.join(mp, "test_summary", "tabular_test_summary.csv"))
]
CV_TEST_PER_SUBJECT_SUMMARY_FILES = [
    os.path.join(mp, "test_summary", "tabular_per_subject_test_summary.csv")
    for mp in CV_REPETITION_FOLDERS
    if os.path.isfile(
        os.path.join(mp, "test_summary", "tabular_per_subject_test_summary.csv")
    )
]

print(f"Found {len(CV_REPETITION_FOLDERS)} repeated cross validation runs.")
print(f"Found {len(CV_TEST_SUMMARY_FILES)} summary test files (slice level).")
print(
    f"Found {len(CV_TEST_PER_SUBJECT_SUMMARY_FILES)} summary test files (per subject)."
)

# %% GATHER ALL VALUES (SLICE LEVEL SUMMARY)

# Initialize an empty list to store the dataframes for each repetition
dfs_slice_level, dfs_patient_level = [], []

# Loop through the CSV files and append them to the list of dataframes
for csv_file in CV_TEST_SUMMARY_FILES:
    df = pd.read_csv(csv_file)
    dfs_slice_level.append(df)

for csv_file in CV_TEST_PER_SUBJECT_SUMMARY_FILES:
    df = pd.read_csv(csv_file)
    dfs_patient_level.append(df)

# Concatenate the dataframes into one
df_all_slice_level = pd.concat(dfs_slice_level, ignore_index=True)
df_all_patient_level = pd.concat(dfs_patient_level, ignore_index=True)

# Save the aggregated results to a CSV file
df_all_slice_level.to_csv(
    os.path.join(args_dict["SAVE_PATH"], "results_aggregated_per_slice.csv"),
    index=False,
)
df_all_patient_level.to_csv(
    os.path.join(args_dict["SAVE_PATH"], "results_aggregated_per_patient.csv"),
    index=False,
)


# %% COMPUTE STATS ON SLICE LEVEL

# Group the data by model version and classification type
grouped_not_ensembled = df_all_slice_level.loc[
    (df_all_slice_level["fold"] != "ensemble")
].groupby(["model_type"])
grouped_ensembled = df_all_slice_level.loc[
    (df_all_slice_level["fold"] == "ensemble")
].groupby(["model_type"])

# Calculate the mean and standard deviation for the relevant performance metrics
metrics = [
    "precision",
    "recall",
    "accuracy",
    "f1-score",
    "auc",
    "matthews_correlation_coefficient",
]
for g_df, df_name in zip(
    [grouped_not_ensembled, grouped_ensembled],
    ["folds_summary_mean_std", "ensemble_summary_mean_std"],
):
    mean = g_df[metrics].mean().reset_index()
    std = g_df[metrics].std().reset_index()

    # add column with stat
    mean.insert(1, "stat", "mean")
    std.insert(1, "stat", "std")

    # Merge the mean and standard deviation dataframes
    mean_std = pd.concat([mean, std], ignore_index=True)

    # Save the mean and standard deviation results to a CSV file
    mean_std.to_csv(
        os.path.join(args_dict["SAVE_PATH"], f"{df_name}_per_slice.csv"), index=False
    )


# %% COMPUTE STATS ON patient LEVEL

# Group the data by model version and classification type
grouped = df_all_patient_level.groupby(["aggregation_method", "model_type"])

# Calculate the mean and standard deviation for the relevant performance metrics
metrics = [
    "precision",
    "recall",
    "accuracy",
    "f1-score",
    "auc",
    "matthews_correlation_coefficient",
]

mean = grouped[metrics].mean().reset_index()
std = grouped[metrics].std().reset_index()

# add column with stat
mean.insert(1, "stat", "mean")
std.insert(1, "stat", "std")

# Merge the mean and standard deviation dataframes
mean_std = pd.concat([mean, std], ignore_index=True)

# Save the mean and standard deviation results to a CSV file
mean_std.to_csv(
    os.path.join(args_dict["SAVE_PATH"], f"{df_name}_per_SUBJECT.csv"), index=False
)
