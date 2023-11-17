# %%
"""
Script runs model evaluation over cross validation models.
The script uses a hydra configuration file for specifying where the models are located, where the dataset is and how is splits, etc.
The script iterates over the models and performs predictions on the subjects specified by the set parameter in the hydra configuration file 
(the set can be training, validation or testing). The predicted logits for each slide is saved into a pandas tadaframe that holds the infromation
abour that slide (subject ID, target class, file_path, etc.). Using this dataframe, the predicted logits are maniputated to obtain:
- per slice ensemble prediction: mean over the logits from the cross-validation models (per_slice_ensemble)
- per slice entropy: Shannon entropy calculated over the softmax of the cross-validation models (per_slice_entropy)
- per sibject fold-wise ensemble: mean logits predictions over all the slice predictions from one of the cross-validation models (subject_ensemble_pred_fold_N)
- per subject fold-wise entropy: Shannon entropy over all the slice predictions from one of the cross-validation models (subject_entropy_pred_fold_N)
- per subject ensemble: mean of predicted logits for all the slices and models (overall_subject_ensemble)
- per subject entropy: Shannon entropy over all slices predictions from all the models (overall_subject_entropy)

"""

import os
import glob
import csv
import json
import numpy as np
import argparse
import pandas as pd
import importlib
import logging
from pathlib import Path
import time
from sklearn.metrics import matthews_corrcoef
from scipy.special import softmax
from scipy.stats import entropy
import importlib

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as T
import pytorch_lightning as pl

# local imports
import dataset_utilities

# import utilities


def check_given_folders(config):
    # -------------------------------------
    # Check that the given folder exist
    # -------------------------------------
    for folder, fd in zip(
        [
            config["working_folder"],
            config["dataloader_settings"]["path_to_data_split_csv_file"],
            config["model_settings"]["path_to_cross_validation_folder"],
        ],
        [
            "working folder",
            "CSV file",
            "Cross validation folder",
        ],
    ):
        if not os.path.exists(folder):
            raise ValueError(f"{fd.capitalize} not found. Given {folder}.")


def set_up(config: dict):
    # ---------------------------------------------------------------------------
    # Create folder in the working directory where to save evaluation performance
    # ---------------------------------------------------------------------------

    save_path = os.path.join(
        config["working_folder"],
        "validation_results",
        Path(config["model_settings"]["path_to_cross_validation_folder"]).parents[0],
        Path(config["model_settings"]["path_to_cross_validation_folder"]).stem,
        config["dataloader_settings"]["set_to_evaluate"],
    )

    Path(save_path).mkdir(parents=True, exist_ok=True)

    # save the path in the configuration
    config["save_path"] = save_path

    # ---------------------
    # Set GPU to be used
    # ---------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["resources"]["gpu_nbr"])


def run_evaluation(config: dict):
    # GET TEST FILES AND BUILD GENERATOR

    # load .csv file
    dataset_split = pd.read_csv(
        config["dataloader_settings"]["path_to_data_split_csv_file"]
    )

    # GET PATHS TO THE MODELS TO EVALUATE
    MODELS = []
    for f_indx, f in enumerate(
        glob.glob(
            os.path.join(
                config["model_settings"]["path_to_cross_validation_folder"], "TB_*", ""
            )
        )
    ):
        # get paths to model
        aus_dict = {
            "last": None,
        }
        for m, mv, mvn in zip(
            ["last", "best"],
            ["last_model", "best_model_weights"],
            ["last_model", "best_model"],
        ):
            if os.path.isfile(os.path.join(f, mv, mvn)):
                aus_dict[m] = os.path.join(f, mv, mvn)
        MODELS.append(aus_dict)

    # print what has been found
    for idx, m in enumerate(MODELS):
        print(
            f"Fold {idx+1}\n  Last model: {True if m['last'] else None}\n  Best model: {True if m['best'] else None}"
        )

    # GET EVALUATION PERFORMANCE ON A SLICE LEVEL
    # here wecan save all the information in a dataframe so that future manipulations are easy to perform.
    # The dataframe stores the subject_ID, the class, the file name and the performance for each model in columns MF_1, MF_2, etc.
    # What is saved are the logits (not softmax) scores as outputed from the model for each of the classes.

    for cv_f, M in enumerate(MODEL):
        print("\n")
        for mv in ["last", "best"]:
            if M[mv]:
                print(
                    f"    Working on fold {cv_f + 1}, {mv} model version... \r    ",
                    end="",
                )
                # load model
                model = torch.load(M[mv])
                # get files for this model evaluation. This depends on the fold and the set_to_evaluate specified.
                # This is easy when using the .cvs file created during the dataset split during training.
                files_for_inference = dataset_split


# %%  WORKING ON IMPLEMENTATION
importlib.reload(dataset_utilities)

config = {
    "working_folder": os.getcwd(),
    "save_path": None,
    "dataloader_settings": {
        "path_to_data_split_csv_file": "/flush/iulta54/P5-PedMRI_CBTN_v1/trained_model_archive/TESTs_20231106/ResNet18_pretrained_True_frozen_True_0.4_LR_0.0001_BATCH_64_AUGMENTATION_True_OPTIM_sgd_SCHEDULER_exponential_MLPNODES_512_t0928/REPETITION_1/data_split_information.csv",
        "set_to_evaluate": "test",  # this can be training, validation, testing
        "preprocessing_settings": {
            "input_size": [240, 240],
            "img_mean": [0.4451, 0.4262, 0.3959],
            "img_std": [0.2411, 0.2403, 0.2466],
        },
    },
    "model_settings": {
        "path_to_cross_validation_folder": "/flush/iulta54/P5-PedMRI_CBTN_v1/trained_model_archive/TESTs_20231106/ResNet18_pretrained_True_frozen_True_0.4_LR_0.0001_BATCH_64_AUGMENTATION_True_OPTIM_sgd_SCHEDULER_exponential_MLPNODES_512_t0928/REPETITION_1",
    },
    "resources": {"gpu_nbr": 3, "nbr_workers": 15},
}

check_given_folders(config)
set_up(config)

# GET FILES, SET UP LABEL DICTIONARY MAP and the pre-processing transform
# load .csv file
dataset_split = pd.read_csv(
    config["dataloader_settings"]["path_to_data_split_csv_file"]
)
dataset_split = dataset_split.drop(columns=["level_0", "index"])

# get mapping of the classes to one hot encoding
unique_targe_classes = dict.fromkeys(pd.unique(dataset_split["target"]))
one_hot_encodings = torch.nn.functional.one_hot(
    torch.tensor(list(range(len(unique_targe_classes))))
)
one_hot_encodings = [i.type(torch.float32) for i in one_hot_encodings]
# build mapping between class and one hot encoding
target_class_to_one_hot_mapping = dict(zip(unique_targe_classes, one_hot_encodings))

# define the trainsformation
pre_process_transform = T.Compose(
    [
        T.Resize(
            size=list(
                config["dataloader_settings"]["preprocessing_settings"]["input_size"]
            ),
            antialias=True,
        ),
        T.ToTensor(),
        T.Normalize(
            mean=list(
                config["dataloader_settings"]["preprocessing_settings"]["img_mean"]
            ),
            std=list(
                config["dataloader_settings"]["preprocessing_settings"]["img_std"]
            ),
        ),
    ],
)
# GET PATHS TO THE MODELS TO EVALUATE
MODELS = []
for f_indx, f in enumerate(
    glob.glob(
        os.path.join(
            config["model_settings"]["path_to_cross_validation_folder"], "TB_*", ""
        )
    )
):
    # get paths to model
    aus_dict = {"last": None, "best": None}
    for mv in ["last", "best"]:
        if os.path.isfile(os.path.join(f, mv + ".pt")):
            aus_dict[mv] = os.path.join(f, mv + ".pt")
    MODELS.append(aus_dict)

# print what has been found
for idx, m in enumerate(MODELS):
    print(
        f"Fold {idx+1}\n  Last model: {True if m['last'] else None}\n  Best model: {True if m['best'] else None}"
    )

#  %%
# GET EVALUATION PERFORMANCE ON A SLICE LEVEL
# here wecan save all the information in a dataframe so that future manipulations are easy to perform.
# The dataframe stores the subject_ID, the class, the file name and the performance for each model in columns MF_1, MF_2, etc.
# What is saved are the logits (not softmax) scores as outputed from the model for each of the classes.
summary_evaluation = []
for cv_f, M in enumerate(MODELS):
    print("\n")
    for mv in ["last", "best"]:
        if M[mv]:
            # load model
            model = torch.load(M[mv])
            model.eval()

            # get files for this model evaluation. This depends on the fold and the set_to_evaluate specified.
            # This is easy when using the .cvs file created during the dataset split during training.
            files_for_inference = dataset_split.loc[
                dataset_split[f"fold_{cv_f+1}"]
                == config["dataloader_settings"]["set_to_evaluate"]
            ]
            # make torch tensor labels
            labels = [
                target_class_to_one_hot_mapping[c]
                for c in list(files_for_inference["target"])
            ]
            labels_for_df = [list(l.numpy()) for l in labels]
            # add label to the dataframe
            files_for_inference.insert(
                files_for_inference.shape[1], "one_hot_encodig", labels_for_df
            )
            # build generator on these files
            dataset = DataLoader(
                dataset_utilities.PNGDatasetFromFolder(
                    list(files_for_inference["file_path"]),
                    transform=pre_process_transform,
                    labels=labels,
                    return_file_path=False,
                ),
                batch_size=64,
                num_workers=config["resources"]["nbr_workers"],
                shuffle=False,
            )
            # get predictions on these filed
            pred_list = []
            dataset = iter(dataset)
            with torch.no_grad():
                for b in range(len(dataset)):
                    print(
                        f"Working on fold {cv_f + 1}, {mv} model version, batch {b+1}/{len(dataset)} \r",
                        end="",
                    )
                    x, y = next(dataset)
                    pred_list.extend(model(x).to("cpu").numpy().tolist())

            # add the predictions to the dataframe
            files_for_inference.insert(
                files_for_inference.shape[1], f"pred_fold_{cv_f+1}", pred_list
            )
            # save dataframe
            summary_evaluation.append(files_for_inference)
# %% WORK ON THE DATAFRAMES
# stack the dataframes
concatenated_summary_evaluation = pd.concat(
    summary_evaluation, axis=0, ignore_index=True
)

# collapse so that each slice appears only once and the predictions from the different models are nicely organized
collapsed_summary_evaluation = (
    concatenated_summary_evaluation.groupby("file_path").first().reset_index()
)
# set the subject_IDs as indexes
collapsed_summary_evaluation = collapsed_summary_evaluation.set_index("subject_IDs")


# %% WORK ON THE SLICE LEVEL AGGREGATIONS
# ensemble and get per-slice entropy based uncertainty
def ensamble_predictions(pd_slice_row, nbr_predictive_models):
    # this function ensambles the predictions for a single slide in the collased summary evaluation dataframe
    # get all the predictions for this slice
    predictions = [
        pd_slice_row[f"pred_fold_{i+1}"] for i in range(nbr_predictive_models)
    ]
    # ensemble predictions
    ensemble = np.array(predictions).mean(axis=0).tolist()
    return ensemble


def per_slice_per_class_uncertainty(pd_slice_row, nbr_predictive_models):
    # this function ensambles the predictions for a single slide in the collased summary evaluation dataframe
    # get all the predictions for this slice
    predictions = [
        pd_slice_row[f"pred_fold_{i+1}"] for i in range(nbr_predictive_models)
    ]
    # apply softmax to get predicted probabilities (NOTE that these are not real probabilities since they are not calibrated)
    predicted_probabilities = softmax(predictions)
    # compute SHannon entropy class wise
    uncertainty = entropy(predicted_probabilities)
    return uncertainty


# compute ensemble
collapsed_summary_evaluation["per_slice_ensemble"] = collapsed_summary_evaluation.apply(
    lambda x: ensamble_predictions(x, len(MODELS)), axis=1
)
# compute uncertainty
collapsed_summary_evaluation[
    "per_slice_per_class_uncertainty"
] = collapsed_summary_evaluation.apply(
    lambda x: per_slice_per_class_uncertainty(x, len(MODELS)), axis=1
)

# %% WORK ON THE PATIENT LEVEL AGGREGATIONS


def proces_subject_wise_predictions(
    grouped_data, func, use_exixting_col_names: bool = True, prefix: str = "new_value"
):
    processed_data = grouped_data.groupby("subject_IDs").apply(lambda x: func(x))

    if isinstance(processed_data.iloc[0][0], list):
        temp = pd.DataFrame(
            processed_data.apply(pd.Series).values.tolist(), index=processed_data.index
        )
    else:
        temp = pd.DataFrame(
            processed_data.to_frame().apply(pd.Series).values.tolist(),
            index=processed_data.index,
        )
    if use_exixting_col_names:
        col_names = [
            f"{prefix}_{col}" for col in grouped_data if col.startswith("pred_fold_")
        ]
    else:
        col_names = [prefix]
    temp.columns = col_names
    grouped_data = grouped_data.merge(
        temp, how="left", left_on="subject_IDs", right_index=True
    )
    return grouped_data


def slice_wise_model_wise_subject_ensemble(x):
    aus = [
        np.array(list(x[col])).mean(axis=0).tolist()
        for col in x
        if col.startswith("pred_fold_")
    ]
    return aus


def slice_wise_model_wise_subject_entropy(x):
    aus = [
        entropy(softmax(np.array(list(x[col])))).tolist()
        for col in x
        if col.startswith("pred_fold_")
    ]
    return aus


def slice_wise_subject_ensemble(x):
    # get all logits from all folds
    aus = []
    aus.append(
        [
            np.array(list(x[col])).mean(axis=0).tolist()
            for col in x
            if col.startswith("pred_fold_")
        ]
    )
    aus = np.vstack(aus).mean(axis=0).tolist()
    return aus


def slice_wise_subject_entropy(x):
    # get all logits from all folds
    aus = []
    aus.append(
        [
            np.array(list(x[col])).mean(axis=0).tolist()
            for col in x
            if col.startswith("pred_fold_")
        ]
    )
    aus = entropy(softmax(np.vstack(aus))).tolist()
    return aus


collapsed_summary_evaluation = proces_subject_wise_predictions(
    collapsed_summary_evaluation,
    slice_wise_model_wise_subject_ensemble,
    prefix="subject_ensemble",
)
collapsed_summary_evaluation = proces_subject_wise_predictions(
    collapsed_summary_evaluation,
    slice_wise_model_wise_subject_entropy,
    prefix="subject_entropy",
)

collapsed_summary_evaluation = proces_subject_wise_predictions(
    collapsed_summary_evaluation,
    slice_wise_subject_ensemble,
    use_exixting_col_names=False,
    prefix="overall_subject_ensemble",
)

collapsed_summary_evaluation = proces_subject_wise_predictions(
    collapsed_summary_evaluation,
    slice_wise_subject_entropy,
    use_exixting_col_names=False,
    prefix="overall_subject_entropy",
)


# %% PLOT SLICE-WISE RESULTS
import utilities

importlib.reload(utilities)

# plot metrics for each fold separately and for the ensemble
# get the column names to plot

col_names = [
    col for col in collapsed_summary_evaluation if col.startswith("pred_fold_")
]
# add ensemble
col_names.extend(["per_slice_ensemble"])


for col_name in col_names:
    print(f"Working on {col_name}\r", end="")
    # get predictions and ground truth
    Ytest_categorical = np.array(list(collapsed_summary_evaluation["one_hot_encodig"]))
    Ptest_softmax = softmax(list(collapsed_summary_evaluation[col_name]))

    # plot metrics
    utilities.plotConfusionMatrix(
        GT=Ytest_categorical,
        PRED=Ptest_softmax,
        classes=["Not_tumor", "Tumor"]
        if len(one_hot_encodings) == 2
        else (
            ["ASTR", "EP", "MED"]
            if len(one_hot_encodings) == 3
            else ["ASTR_in", "ASTR_su", "EP_in", "EP_su", "MED_in"]
        ),
        savePath=config["save_path"],
        saveName=f"slice_level_CM_{mv}_model_{col_name}",
        draw=False,
    )

    utilities.plotROC(
        GT=Ytest_categorical,
        PRED=Ptest_softmax,
        classes=["Not_tumor", "Tumor"]
        if len(one_hot_encodings) == 2
        else (
            ["ASTR", "EP", "MED"]
            if len(one_hot_encodings) == 3
            else ["ASTR_in", "ASTR_su", "EP_in", "EP_su", "MED_in"]
        ),
        savePath=config["save_path"],
        saveName=f"slice_level_ROC_{mv}_model_fold_{col_name}",
        draw=False,
    )

    utilities.plotPR(
        GT=Ytest_categorical,
        PRED=Ptest_softmax,
        classes=["Not_tumor", "Tumor"]
        if len(one_hot_encodings) == 2
        else (
            ["ASTR", "EP", "MED"]
            if len(one_hot_encodings) == 3
            else ["ASTR_in", "ASTR_su", "EP_in", "EP_su", "MED_in"]
        ),
        savePath=config["save_path"],
        saveName=f"slice_level_PR_{mv}_model_fold_{col_name}",
        draw=False,
    )

# %% PLOT SUBJECT LEVEL METRICS
col_names = [
    col
    for col in collapsed_summary_evaluation
    if col.startswith("subject_ensemble_pred_")
]
# add ensemble
col_names.extend(["overall_subject_ensemble"])

for col_name in col_names:
    print(f"Working on {col_name}\r", end="")
    # get predictions and ground truth
    Ytest_categorical = np.array(
        list(
            collapsed_summary_evaluation.groupby(collapsed_summary_evaluation.index)[
                "one_hot_encodig"
            ].first()
        )
    )
    Ptest_softmax = softmax(
        list(
            collapsed_summary_evaluation.groupby(collapsed_summary_evaluation.index)[
                col_name
            ].first()
        )
    )

    # plot metrics
    utilities.plotConfusionMatrix(
        GT=Ytest_categorical,
        PRED=Ptest_softmax,
        classes=["Not_tumor", "Tumor"]
        if len(one_hot_encodings) == 2
        else (
            ["ASTR", "EP", "MED"]
            if len(one_hot_encodings) == 3
            else ["ASTR_in", "ASTR_su", "EP_in", "EP_su", "MED_in"]
        ),
        savePath=config["save_path"],
        saveName=f"subject_level_CM_{mv}_model_{col_name}",
        draw=False,
    )

    utilities.plotROC(
        GT=Ytest_categorical,
        PRED=Ptest_softmax,
        classes=["Not_tumor", "Tumor"]
        if len(one_hot_encodings) == 2
        else (
            ["ASTR", "EP", "MED"]
            if len(one_hot_encodings) == 3
            else ["ASTR_in", "ASTR_su", "EP_in", "EP_su", "MED_in"]
        ),
        savePath=config["save_path"],
        saveName=f"subject_level_ROC_{mv}_model_fold_{col_name}",
        draw=False,
    )

    utilities.plotPR(
        GT=Ytest_categorical,
        PRED=Ptest_softmax,
        classes=["Not_tumor", "Tumor"]
        if len(one_hot_encodings) == 2
        else (
            ["ASTR", "EP", "MED"]
            if len(one_hot_encodings) == 3
            else ["ASTR_in", "ASTR_su", "EP_in", "EP_su", "MED_in"]
        ),
        savePath=config["save_path"],
        saveName=f"subject_level_PR_{mv}_model_fold_{col_name}",
        draw=False,
    )
# %%  COMPUTE METRICS AND SAVE IN THE DATAFRAME
#TODO
# %% SAVE PANDAS TO FILE

collapsed_summary_evaluation.to_csv(
    os.path.join(config["save_path"], "full_evaluation.csv")
)

# %%
# @hydra.main(version_base=None, config_path="conf", config_name="config")
# def main(config: DictConfig):
#     config = dict(config)

#     check_given_folders(config)
#     set_up(config)
#     run_evaluation(config)


# if __name__ == "__main__":
#     main()
