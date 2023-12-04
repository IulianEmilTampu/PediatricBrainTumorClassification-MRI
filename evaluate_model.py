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
import sys
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
from copy import deepcopy

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as T
import pytorch_lightning as pl

# local imports
import dataset_utilities

pl.seed_everything(42)


# %%
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
            raise ValueError(f"{fd} not found. Given {folder}.")


def set_up(evaluation_config):
    # load the corrisponding training_configuration .yaml file
    try:
        training_config = dict(
            OmegaConf.load(
                os.path.join(
                    evaluation_config["model_settings"][
                        "path_to_cross_validation_folder"
                    ],
                    "hydra_config.yaml",
                )
            )
        )
    except:
        raise ValueError(
            f"Training configuration .yaml file not found. Given {evaluation_config['model_settings']['path_to_cross_validation_folder']}"
        )

    # -----------------------------------------------------------------------------
    # Build evaluation configuration dictionary based on the training configuration
    # -----------------------------------------------------------------------------
    # just for convenience
    aus_repetition_path = evaluation_config["model_settings"][
        "path_to_cross_validation_folder"
    ]

    evaluation_config["working_folder"] = (
        os.getcwd()
        if not evaluation_config["working_folder"]
        else evaluation_config["working_folder"]
    )

    evaluation_config["save_path"] = os.path.join(
        aus_repetition_path,
        "evaluation_results",
        evaluation_config["dataloader_settings"]["set_to_evaluate"],
    )

    evaluation_config["model_settings"]["model_version"] = str(
        os.path.basename(os.path.dirname(Path(aus_repetition_path)))
    ).split("_")[0]

    evaluation_config["model_settings"]["session_time"] = str(
        os.path.basename(os.path.dirname(Path(aus_repetition_path)))
    ).split("_")[-1]

    evaluation_config["model_settings"]["repetition_nbr"] = str(
        os.path.basename(Path(aus_repetition_path))
    )

    evaluation_config["dataloader_settings"][
        "path_to_data_split_csv_file"
    ] = os.path.join(aus_repetition_path, "data_split_information.csv")

    evaluation_config["dataloader_settings"]["preprocessing_settings"]["input_size"] = (
        training_config["dataloader_settings"]["input_size"]
        if not evaluation_config["dataloader_settings"]["preprocessing_settings"][
            "input_size"
        ]
        else evaluation_config["dataloader_settings"]["preprocessing_settings"][
            "input_size"
        ]
    )

    evaluation_config["dataloader_settings"]["preprocessing_settings"]["img_mean"] = (
        training_config["dataloader_settings"]["img_mean"]
        if not evaluation_config["dataloader_settings"]["preprocessing_settings"][
            "img_mean"
        ]
        else evaluation_config["dataloader_settings"]["preprocessing_settings"][
            "img_mean"
        ]
    )
    evaluation_config["dataloader_settings"]["preprocessing_settings"]["img_std"] = (
        training_config["dataloader_settings"]["img_std"]
        if not evaluation_config["dataloader_settings"]["preprocessing_settings"][
            "img_std"
        ]
        else evaluation_config["dataloader_settings"]["preprocessing_settings"][
            "img_std"
        ]
    )

    evaluation_config["dataloader_settings"]["use_age"] = (
        training_config["dataloader_settings"]["use_age"]
        if "use_age" in training_config["dataloader_settings"].keys()
        else False
    )
    evaluation_config["dataloader_settings"]["normalize_age"] = (
        training_config["dataloader_settings"]["normalize_age"]
        if "normalize_age" in training_config["dataloader_settings"].keys()
        else False
    )

    # ------------------------
    # Run check on the folder
    # -----------------------
    check_given_folders(evaluation_config)

    # ---------------------------------------------------------------------------
    # Create folder in the working directory where to save evaluation performance
    # ---------------------------------------------------------------------------
    Path(evaluation_config["save_path"]).mkdir(parents=True, exist_ok=True)

    # ---------------------
    # Set GPU to be used
    # ---------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(evaluation_config["resources"]["gpu_nbr"])

    return evaluation_config


def run_evaluation_repetition_evaluation(config: dict):
    # %%GET FILES, SET UP LABEL DICTIONARY MAP and the pre-processing transform
    # load .csv file
    print(
        f'Dataset split path: {f"{os.path.sep}".join((Path(config["dataloader_settings"]["path_to_data_split_csv_file"]).parts[-3::]))}'
    )
    dataset_split = pd.read_csv(
        config["dataloader_settings"]["path_to_data_split_csv_file"]
    )
    try:
        dataset_split = dataset_split.drop(columns=["level_0", "index"])
    except:
        print()

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
                    config["dataloader_settings"]["preprocessing_settings"][
                        "input_size"
                    ]
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
        aus_dict = {"last": None, "best": None, "fold_nbr": None}
        for mv in ["last", "best"]:
            if os.path.isfile(os.path.join(f, mv + ".pt")):
                aus_dict[mv] = os.path.join(f, mv + ".pt")
        aus_dict["fold_nbr"] = int(os.path.basename(Path(f)).split("_")[-1])
        MODELS.append(aus_dict)

    # print what has been found
    for idx, m in enumerate(MODELS):
        print(
            f"Fold {m['fold_nbr']}\n  Last model: {True if m['last'] else None}\n  Best model: {True if m['best'] else None}"
        )

    #  %% GET EVALUATION PERFORMANCE ON A SLICE LEVEL
    # Here we can save all the information in a dataframe so that future manipulations are easy to perform.
    # The dataframe stores the subject_ID, the class, the file name and the performance for each model in columns MF_1, MF_2, etc.
    # What is saved are the logits (not softmax) scores as outputed from the model for each of the classes.
    summary_evaluation = []
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    for M in MODELS:
        print("\n")
        for mv in ["last", "best"]:
            if M[mv]:
                # load model
                model = torch.load(M[mv])
                model.to(device)
                model.eval()

                # get files for this model evaluation. This depends on the fold and the set_to_evaluate specified.
                # This is easy when using the .cvs file created during the dataset split during training.
                files_for_inference = dataset_split.loc[
                    dataset_split[f"fold_{M['fold_nbr']}"]
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
                        return_age=config["dataloader_settings"]["use_age"],
                        ages=(
                            list(files_for_inference["age_normalized"])
                            if config["dataloader_settings"]["normalize_age"]
                            else list(files_for_inference["age_in_days"])
                        )
                        if config["dataloader_settings"]["use_age"]
                        else None,
                    ),
                    batch_size=32,
                    num_workers=15,
                    shuffle=False,
                )
                # get predictions on these filed
                pred_list = []
                dataset = iter(dataset)
                with torch.no_grad():
                    for b in range(len(dataset)):
                        print(
                            f"Working on fold {M['fold_nbr']}, {mv} model version, batch {b+1}/{len(dataset)} \r",
                            end="",
                        )
                        batch = next(dataset)
                        if config["dataloader_settings"]["use_age"]:
                            x = (batch[0].to(device), batch[2].to(device))
                            y = batch[1]
                        else:
                            x = batch[0].to(device)
                            y = batch[1]
                        pred_list.extend(model(x).to("cpu").numpy().tolist())

                # add the predictions to the dataframe
                files_for_inference.insert(
                    files_for_inference.shape[1],
                    f"pred_fold_{M['fold_nbr']}",
                    pred_list,
                )
                # save dataframe
                summary_evaluation.append(files_for_inference)
    print("\n")

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
    def ensamble_predictions(pd_slice_row, nbr_predictive_models):
        # this function ensambles the predictions for a single slide over the different folds in the collased summary evaluation dataframe

        # get all the predictions for this slice
        predictions = [
            pd_slice_row[f"pred_fold_{i+1}"] for i in range(nbr_predictive_models)
        ]

        # remove None
        predictions = [x for x in predictions if x is not None]
        # ensemble predictions
        ensemble = np.array(predictions).mean(axis=0).tolist()
        return ensemble

    def per_slice_per_class_uncertainty(pd_slice_row, nbr_predictive_models):
        # this function ensambles the predictions for a single slide over the different folds in the collased summary evaluation dataframe

        # get all the predictions for this slice
        predictions = [
            pd_slice_row[f"pred_fold_{i+1}"] for i in range(nbr_predictive_models)
        ]
        # remove None
        predictions = [x for x in predictions if x is not None]
        # apply softmax to get predicted probabilities (NOTE that these are not real probabilities since they are not calibrated)
        predicted_probabilities = softmax(predictions)
        # compute SHannon entropy class wise
        uncertainty = entropy(predicted_probabilities)
        return uncertainty

    # compute ensemble
    collapsed_summary_evaluation[
        "per_slice_ensemble"
    ] = collapsed_summary_evaluation.apply(
        lambda x: ensamble_predictions(x, len(MODELS)), axis=1
    )
    # compute uncertainty
    collapsed_summary_evaluation[
        "per_slice_per_class_uncertainty"
    ] = collapsed_summary_evaluation.apply(
        lambda x: per_slice_per_class_uncertainty(x, len(MODELS)), axis=1
    )

    # %% WORK ON THE PATIENT LEVEL AGGREGATIONS
    # This is a little bit nasty since, in the case of thraining and validation, there are subjects that are only used in one of the folds.
    # Thus, need to account for the None values and substitute such that the plotting and following code works.
    def proces_subject_wise_predictions(
        grouped_data,
        func,
        use_exixting_col_names: bool = True,
        prefix: str = "new_value",
    ):
        processed_data = grouped_data.groupby("subject_IDs").apply(lambda x: func(x))

        if isinstance(processed_data.iloc[0][0], list):
            temp = pd.DataFrame(
                processed_data.apply(pd.Series).values.tolist(),
                index=processed_data.index,
            )
        else:
            temp = pd.DataFrame(
                processed_data.to_frame().apply(pd.Series).values.tolist(),
                index=processed_data.index,
            )

        if use_exixting_col_names:
            col_names = [
                f"{prefix}_{col}"
                for col in grouped_data
                if col.startswith("pred_fold_")
            ]
        else:
            col_names = [prefix]
        temp.columns = col_names

        grouped_data = grouped_data.merge(
            temp, how="left", left_on="subject_IDs", right_index=True
        )
        return grouped_data

    # ensemble of the predictions for a given slice
    def slice_wise_model_wise_subject_ensemble(x):
        # here need to handle the cases where there are None values (e.g. subject only use in one fold)
        aus = []
        for col in x:
            if col.startswith("pred_fold_"):
                if x[col][0] != None:
                    aus.append(np.array(list(x[col])).mean(axis=0).tolist())
                else:
                    aus.append([None])
        return aus

    def slice_wise_model_wise_subject_entropy(x):
        # here need to handle the cases where there are None values (e.g. subject only use in one fold)
        aus = []
        for col in x:
            if col.startswith("pred_fold_"):
                if x[col][0] != None:
                    aus.append(entropy(softmax(np.array(list(x[col])))).tolist())
                else:
                    # if it is None
                    aus.append([None])
        return aus

    def slice_wise_subject_ensemble(x):
        # here need to handle the cases where there are None values (e.g. subject only use in one fold)
        aus = []
        for col in x:
            if col.startswith("pred_fold_"):
                if x[col][0] != None:
                    aus.append(np.array(list(x[col])).mean(axis=0).tolist())
                # else:
                #     # if it is None
                #     aus.append([None,None,None])
        aus = np.vstack(aus).mean(axis=0).tolist()
        return aus

    def slice_wise_subject_entropy(x):
        # here need to handle the cases where there are None values (e.g. subject only use in one fold)
        aus = []
        for col in x:
            if col.startswith("pred_fold_"):
                if x[col][0] != None:
                    aus.append(np.array(list(x[col])).mean(axis=0).tolist())
                # else:
                #     # if it is None
                #     aus.append([0, 0, 0])
        # compute entropy
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

    # %% INITIATE DICT FOR SUMMARY PERFOMANCE TO CSV
    # this holds the dictionaries from the evaluation_utilities.get_performance_metrics for all the configurations:
    # - per fold slice wise
    # - per slice ensemble across folds
    # - per subject slice wise
    # -per subject across folds and slices ensembles
    summary_performance = []
    # %% PLOT SLICE-WISE RESULTS
    import evaluation_utilities

    importlib.reload(evaluation_utilities)

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
        Ytest_categorical = np.array(
            list(collapsed_summary_evaluation["one_hot_encodig"])
        )

        # remove nones from the predicitions
        Ptest = list(collapsed_summary_evaluation[col_name])
        idx_not_none = [i for i, _ in enumerate(Ptest) if Ptest[i] != None]
        Ptest = np.array([Ptest[i] for i in idx_not_none])
        Ytest_categorical = np.array([Ytest_categorical[i] for i in idx_not_none])

        Ptest_softmax = softmax(np.array(Ptest), axis=1)

        # get metrics
        aus_dict = evaluation_utilities.get_performance_metrics(
            Ytest_categorical, Ptest_softmax
        )

        # add general information
        aus_dict["model_version"] = "_".join(
            [
                config["model_settings"]["model_version"],
                config["model_settings"]["session_time"],
            ]
        )
        aus_dict["repetition"] = config["model_settings"]["repetition_nbr"]
        aus_dict["nbr_classes"] = len(list(unique_targe_classes.keys()))
        aus_dict["classes"] = list(unique_targe_classes.keys())
        aus_dict["dataset_version"] = (
            "full" if dataset_split.shape[0] > 1500 else "[45,55]"
        )
        aus_dict["fold_nbr"] = (
            "all" if col_name == "per_slice_ensemble" else int(col_name.split("_")[-1])
        )
        aus_dict["performance_over"] = col_name

        # save
        summary_performance.append(aus_dict)

        # plot metrics
        evaluation_utilities.plotConfusionMatrix(
            GT=Ytest_categorical,
            PRED=Ptest_softmax,
            classes=list(unique_targe_classes.keys()),
            savePath=config["save_path"],
            saveName=f"slice_level_CM_{'last'}_model_{col_name}",
            draw=False,
        )

        evaluation_utilities.plotROC(
            GT=Ytest_categorical,
            PRED=Ptest_softmax,
            classes=list(unique_targe_classes.keys()),
            savePath=config["save_path"],
            saveName=f"slice_level_ROC_{'last'}_model_fold_{col_name}",
            draw=False,
        )

        evaluation_utilities.plotPR(
            GT=Ytest_categorical,
            PRED=Ptest_softmax,
            classes=list(unique_targe_classes.keys()),
            savePath=config["save_path"],
            saveName=f"slice_level_PR_{'last'}_model_fold_{col_name}",
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
                collapsed_summary_evaluation.groupby(
                    collapsed_summary_evaluation.index
                )["one_hot_encodig"].first()
            )
        )

        Ptest = list(
            collapsed_summary_evaluation.groupby(collapsed_summary_evaluation.index)[
                col_name
            ].first()
        )

        idx_not_none = [i for i, _ in enumerate(Ptest) if Ptest[i] != [None]]
        Ptest = np.array([Ptest[i] for i in idx_not_none])
        Ytest_categorical = np.array([Ytest_categorical[i] for i in idx_not_none])

        Ptest_softmax = softmax(np.array(Ptest), axis=1)

        # get metrics
        aus_dict = evaluation_utilities.get_performance_metrics(
            Ytest_categorical, Ptest_softmax
        )

        # add general information
        aus_dict["model_version"] = "_".join(
            [
                config["model_settings"]["model_version"],
                config["model_settings"]["session_time"],
            ]
        )
        aus_dict["repetition"] = config["model_settings"]["repetition_nbr"]
        aus_dict["nbr_classes"] = len(list(unique_targe_classes.keys()))
        aus_dict["classes"] = list(unique_targe_classes.keys())
        aus_dict["dataset_version"] = (
            "full" if dataset_split.shape[0] > 1500 else "[45,55]"
        )
        aus_dict["fold_nbr"] = (
            "all"
            if col_name == "overall_subject_ensemble"
            else int(col_name.split("_")[-1])
        )
        aus_dict["performance_over"] = col_name

        # save
        summary_performance.append(aus_dict)

        # plot metrics
        evaluation_utilities.plotConfusionMatrix(
            GT=Ytest_categorical,
            PRED=Ptest_softmax,
            classes=list(unique_targe_classes.keys()),
            savePath=config["save_path"],
            saveName=f"subject_level_CM_{mv}_model_{col_name}",
            draw=False,
        )

        evaluation_utilities.plotROC(
            GT=Ytest_categorical,
            PRED=Ptest_softmax,
            classes=list(unique_targe_classes.keys()),
            savePath=config["save_path"],
            saveName=f"subject_level_ROC_{mv}_model_fold_{col_name}",
            draw=False,
        )

        evaluation_utilities.plotPR(
            GT=Ytest_categorical,
            PRED=Ptest_softmax,
            classes=list(unique_targe_classes.keys()),
            savePath=config["save_path"],
            saveName=f"subject_level_PR_{mv}_model_fold_{col_name}",
            draw=False,
        )

    # %% SAVE PANDAS TO FILEs

    collapsed_summary_evaluation.to_csv(
        os.path.join(config["save_path"], "full_evaluation.csv")
    )

    # save also performance
    df = pd.DataFrame(summary_performance)
    df.to_csv(os.path.join(config["save_path"], "summary_performance.csv"))


# %% MAIN
@hydra.main(version_base=None, config_path="conf", config_name="evaluation_config")
def main(evaluation_config: DictConfig):
    evaluation_config = dict(evaluation_config)

    # get repetion folder
    repetitions_to_evaluate = []
    for repetition in glob.glob(
        os.path.join(
            evaluation_config["path_to_repetition_folders"],
            "REPETITION_*",
            "",
        )
    ):
        repetitions_to_evaluate.append(repetition)
    print(f"Found {len(repetitions_to_evaluate)} repetitions to work on...")

    # run for all the repetitions
    for repetition_folder in repetitions_to_evaluate:
        evaluation_config["model_settings"][
            "path_to_cross_validation_folder"
        ] = repetition_folder

        evaluation_config = set_up(evaluation_config)
        print(
            f'Worning on {evaluation_config["model_settings"]["path_to_cross_validation_folder"]}'
        )
        run_evaluation_repetition_evaluation(evaluation_config)


if __name__ == "__main__":
    main()
