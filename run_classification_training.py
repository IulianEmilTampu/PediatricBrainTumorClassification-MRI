# %% IMPORTS
import os
import sys
import glob
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import PIL
from datetime import datetime
import logging
import seaborn as sns

from sklearn.model_selection import (
    KFold,
    train_test_split,
    StratifiedKFold,
    StratifiedGroupKFold,
    GroupShuffleSplit,
    StratifiedShuffleSplit,
)
from sklearn.utils import class_weight
import scipy.stats as stats

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as T

from torchsummary import summary

import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

"""
Some useful links 
https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
"""

# local imports
import core_utils.model_utilities
import core_utils.dataset_utilities


def set_up(config: dict):
    # -------------------------------------
    # Start logger in the SAVE_PATH folder
    # -------------------------------------
    logging.basicConfig(
        filename=os.path.join(
            config["logging_settings"]["checkpoint_path"], "logs.txt"
        ),
        level=logging.DEBUG,
        format="pedMRI - %(asctime)s - %(message)s",
    )
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    # logging.getLogger().addHandler(console)

    # define logger for this application
    setup_logger = logging.getLogger("main.setup")

    setup_logger.info(f" ### STARTING JOB ### ")
    setup_logger.info(
        f'Saving job information at {config["logging_settings"]["checkpoint_path"]}'
    )

    # ---------------------
    # Set GPU to be used
    # ---------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["resources"]["gpu_nbr"])

    # -------------------------------------------
    # CHECK IF RUNNING IN 'RESTART TRAINING' MODE
    # -------------------------------------------

    if all(
        [
            "restart_training_settings" in config.keys(),
            config["restart_training_settings"]["restart_training"],
        ]
    ):
        setup_logger.info("##### RUNNING IN RESTART TRAINING MODE #####")
        if not os.path.isdir(config["restart_training_settings"]["model_folder_path"]):
            raise ValueError(
                f'Requesting script to run in restart training mode, but the given model path does not exist. Given {config["restart_training_settings"]["model_folder_path"]}'
            )

        """
        Need to check how far the model went, and set restart_from.
        - load the configuration file for this run.
        - check how many repetition and cross validation folds were expected
        - get how far it went by checking the repetition and fold folders (there must be a last.pt file for a fold to be complete).
        - set restart_from:
            - it is a dict with as many keys as the total repetitions and for each the folds that need to be run:
                e.g. -> {0: [], 1: [], 2:[3,4], 3:[0,1,2,3,4]} this means that the model will resume training from 
                        repetition 2, running fold 2 and 3, and then repetition 3 all the folds.
        """
        # load the stopped training configuration file
        stopped_training_config = OmegaConf.load(
            os.path.join(
                config["restart_training_settings"]["model_folder_path"],
                "REPETITION_1",
                "hydra_config.yaml",
            )
        )
        # get information from this configuration file
        nbr_expected_repetitions = (
            stopped_training_config.training_settings.nbr_repetitions
        )
        nbr_expected_folds = (
            stopped_training_config.training_settings.nbr_inner_cv_folds
        )
        setup_logger.info(
            f"  Expected {nbr_expected_repetitions} repetitions, each with {nbr_expected_folds} folds."
        )
        # build restart_from
        restart_from = dict.fromkeys(range(nbr_expected_repetitions))

        # loop through the repetition folders and check if all the folds where completed
        for rep in range(nbr_expected_repetitions):
            rep_folder = os.path.join(
                config["restart_training_settings"]["model_folder_path"],
                f"REPETITION_{rep+1}",
            )
            if not os.path.isdir(rep_folder):
                # flag all the folds for this repetition to be trained
                restart_from[rep] = list(range(nbr_expected_folds))
            else:
                restart_from[rep] = []
                # this repetition has been initiated. Check how many folds it has completed
                for fold in range(nbr_expected_folds):
                    fold_folder = os.path.join(rep_folder, f"TB_fold_{fold+1}")
                    if not os.path.isdir(fold_folder):
                        # flag this fold for training
                        restart_from[rep].append(fold)
                    else:
                        # this fold has been initiated, check if it has finished.
                        last_model_path = os.path.join(fold_folder, "last.pt")
                        if not os.path.isfile(last_model_path):
                            # the last model was not saved, meaning that the fold was not completed.
                            restart_from[rep].append(fold)
        # print summary of what is going to be trained
        for rep, value in restart_from.items():
            if len(value) == nbr_expected_folds:
                setup_logger.info(
                    f"  Repetition {rep+1:0{len(str(nbr_expected_repetitions))}d}: running all the folds."
                )
            else:
                setup_logger.info(
                    f"  Repetition {rep+1:0{len(str(nbr_expected_repetitions))}d}: running folds {[v+1 for v in value]}."
                )

        # fix path to point to the local machine
        # ## pretraining model
        if config["working_dir"] != stopped_training_config.working_dir:
            stopped_training_config.working_dir = config["working_dir"]
            if stopped_training_config.model_settings.use_SimCLR_pretrained_model:
                old_path = (
                    stopped_training_config.model_settings.SimCLR_prettrained_model_setitngs.model_path
                )
                if (
                    stopped_training_config.model_settings.SimCLR_prettrained_model_setitngs.pretraining_dataset
                    == "cbtn"
                ):
                    new_path = os.path.join(
                        config["restart_training_settings"][
                            "local_path_to_pretrained_models"
                        ],
                        "CBTN",
                        pathlib.Path(old_path).parts[-1],
                    )
                else:
                    new_path = os.path.join(
                        config["restart_training_settings"][
                            "local_path_to_pretrained_models"
                        ],
                        "TCGA",
                        os.path.join(*pathlib.Path(old_path).parts[-4]),
                    )
                stopped_training_config.model_settings.SimCLR_prettrained_model_setitngs.model_path = (
                    new_path
                )

            # ## dataset path
            print("Changing file path to match the local dataset location.")
            # need to re-map the file paths to match the dataset location on this computer
            # Get dataset information from the dataset_information field.

            classes_long_name_to_short = {
                "ASTROCYTOMA": "ASTR",
                "EPENDYMOMA": "EP",
                "MEDULLOBLASTOMA": "MED",
            }

            classes_string = "_".join(
                [
                    classes_long_name_to_short[c]
                    for c in list(
                        stopped_training_config["dataset"]["classes_of_interest"]
                    )
                ]
            )
            path_to_local_yaml_dataset_configuration_file = os.path.join(
                config["restart_training_settings"]["local_path_to_dataset_configs"],
                "_".join(
                    [
                        stopped_training_config.dataset.modality,
                        stopped_training_config.dataset.name,
                        classes_string,
                    ]
                )
                + ".yaml",
            )
            # load local.yaml and save local dataset path
            local_dataset_config = OmegaConf.load(
                path_to_local_yaml_dataset_configuration_file
            )

            stopped_training_config.dataset.dataset_path = (
                local_dataset_config.dataset_path
            )

        # convert the stopped_training_config into a dict and save restart_from
        stopped_training_config = dict(stopped_training_config)
        # add the restart configuration
        stopped_training_config["restart_training_settings"] = config[
            "restart_training_settings"
        ]
        # and what to restart
        stopped_training_config["restart_training_settings"][
            "restart_from"
        ] = restart_from
        # replace configuration file with the old one
        config = stopped_training_config

    else:
        # -------------------------------------
        # Create folder where to save the model
        # -------------------------------------
        config["logging_settings"]["checkpoint_path"] = os.path.join(
            config["logging_settings"]["checkpoint_path"],
            f"{datetime.now().strftime('%Y%m%d')}",
        )

        Path(config["logging_settings"]["checkpoint_path"]).mkdir(
            parents=True, exist_ok=True
        )

        # save starting day and time
        # OmegaConf.set_struct(config, True)
        # with open_dict(config):
        config["logging_settings"]["start_day"] = datetime.now().strftime("%Y%m%d")
        config["logging_settings"]["start_time"] = datetime.now().strftime("t%H%M%S")

    # ---------------------
    # SEED EVERYTHING
    # ---------------------
    pl.seed_everything(
        config["training_settings"]["random_state"]
    )  # To be reproducable

    return config


def get_info_from_SimCLR_CBTN_pretraining(config: dict):
    """
    Utility that scrapes the SimCLR pretraining information to make it available for
    the classification training.
    In the case the pretraining dataset is config["model_settings"]["SimCLR_prettrained_model_setitngs"]["pretraining_dataset"] == CBTN
    if returns the paths to the dataset splits used for the pretraining so that we avoid data leakege.
    """
    logger = logging.getLogger("main.get_info_from_SimCLR_pretraining")
    logger.info(
        "Loading datasplit information from file (SimCLR pretraining on CBTN data)."
    )

    # loop though the repetition folders and get the datasplit .csv file and model paths.
    info_from_simclr_pretraining = {}
    repetition_folders = glob.glob(
        os.path.join(
            config["model_settings"]["SimCLR_prettrained_model_setitngs"]["model_path"],
            "REPETITION_*",
        )
    )
    for rep in repetition_folders:
        # get repetition number
        rep_nbr = int(os.path.basename(pathlib.Path(rep)).split("_")[-1])
        info_from_simclr_pretraining[rep_nbr] = {
            "datasplit_path": os.path.join(rep, "data_split_information.csv"),
            "per_fold_model_paths": {},
            "hydra_config_file": os.path.join(rep, "hydra_config.yaml"),
        }
        # loop throught the folds and get the path to the pre-trained model
        fold_folders = glob.glob(os.path.join(rep, "TB_fold_*"))
        for fold in fold_folders:
            # get fold number
            fold_nbr = int(os.path.basename(pathlib.Path(fold)).split("_")[-1])
            info_from_simclr_pretraining[rep_nbr]["per_fold_model_paths"][fold_nbr] = (
                os.path.join(fold, "last.pt")
            )

    # add the pretraining model version information
    # load the SimCLR pre-training config file and get information
    simclr_pretraining_config = OmegaConf.load(
        info_from_simclr_pretraining[1]["hydra_config_file"]
    )
    # raise an error if the model version for the SimCLR and the classifier are different
    if (
        simclr_pretraining_config.model_settings.model_version.lower()
        != config["model_settings"]["model_version"].lower()
    ):
        raise ValueError(
            f'The SimCLR pretrained model version and the set classified model version are different. Given (SimCLR) {info_from_simclr_pretraining["pre_trained_model_version"].lower()} != (classifier) {config["model_settings"]["model_version"].lower()}'
        )

    # adjust the config valus for the number of folds and repetitions
    nbr_folds = len(fold_folders)
    config["training_settings"]["nbr_inner_cv_folds"] = nbr_folds
    config["training_settings"]["nbr_repetitions"] = len(repetition_folders)
    logger.info(
        f"Resetting the number of inner folds ({nbr_folds}) and repetitions ({len(repetition_folders)}) using the infromation from the SimCLR pretraining folder."
    )

    return info_from_simclr_pretraining


def run_training(config: dict) -> None:
    logger = logging.getLogger("main.run_training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  load HP from configyaml file (just for convenience)
    BATCH_SIZE = config["dataloader_settings"]["batch_size"]
    LEARNING_RATE = config["training_settings"]["learning_rate"]
    MODEL = config["model_settings"]["model_version"]
    INPUT_SIZE = list(config["dataloader_settings"]["input_size"])
    # pretraining
    PRETRAINED = config["model_settings"]["pre_trained"]
    PRETRAINED_TYPE = (
        "ImageNet"
        if any(
            [
                not config["model_settings"]["use_SimCLR_pretrained_model"],
                "use_SimCLR_pretrained_model" not in config["model_settings"].keys(),
            ]
        )
        else "SimCLR"
    )
    PRETRAINED_DATASET = (
        "ImageNet"
        if PRETRAINED_TYPE == "ImageNet"
        else config["model_settings"]["SimCLR_prettrained_model_setitngs"][
            "pretraining_dataset"
        ].upper()
    )

    FROZE_WEIGHTS = config["model_settings"]["freeze_weights"]
    PERCENTAGE_FROZEN = config["model_settings"]["percentage_freeze_weights"]

    # %% DATA GENERATOR
    # ## define mean and std for normalization
    ImageNet_ResNet_mean = np.array([0.4451, 0.4262, 0.3959])
    ImageNet_ResNet_std = np.array([0.2411, 0.2403, 0.2466])

    ImageNet_VGG_mean = np.array([0.485, 0.456, 0.406])
    ImageNet_VGG_std = np.array([0.229, 0.224, 0.225])

    MR_mean = np.array([0.5, 0.5, 0.5])
    MR_std = np.array([0.5, 0.5, 0.5])

    # select the mean and std for the normalization
    if PRETRAINED:
        if PRETRAINED_TYPE == "ImageNet":
            # use ImageNet stats since the weights were trained using those stats
            img_mean = ImageNet_ResNet_mean
            img_std = ImageNet_ResNet_std
        else:
            # try to load the SimCLR pretraining img_mean and img_std from the pre_training .yaml file
            if os.path.isdir(
                config["model_settings"]["SimCLR_prettrained_model_setitngs"][
                    "model_path"
                ]
            ):
                # here the folder with the repetitions is given.
                # Load the .yaml file from the first repetition.
                pre_training_config = OmegaConf.load(
                    os.path.join(
                        config["model_settings"]["SimCLR_prettrained_model_setitngs"][
                            "model_path"
                        ],
                        "REPETITION_1",
                        "hydra_config.yaml",
                    )
                )
                img_mean = np.array(pre_training_config.dataloader_settings.img_mean)
                img_std = np.array(pre_training_config.dataloader_settings.img_std)
            else:
                # here the path points directly to the model .pt file. Move back to the repeittion folder and load .yaml
                pre_training_config = OmegaConf.load(
                    os.path.join(
                        os.path.dirname(
                            os.path.dirname(
                                config["model_settings"][
                                    "SimCLR_prettrained_model_setitngs"
                                ]["model_path"]
                            )
                        ),
                        "hydra_config.yaml",
                    )
                )
                img_mean = np.array(pre_training_config.dataloader_settings.img_mean)
                img_std = np.array(pre_training_config.dataloader_settings.img_std)
    else:
        # use MR stats since the encoder weights are trained
        img_mean = MR_mean
        img_std = MR_std

    # img_mean = np.array([0.5, 0.5, 0.5])
    # img_std = np.array([0.5, 0.5, 0.5])

    # save image mean and std
    config["dataloader_settings"]["img_mean"] = img_mean.tolist()
    config["dataloader_settings"]["img_std"] = img_std.tolist()

    # ## define preprocessing
    preprocess = T.Compose(
        [
            T.Resize(size=INPUT_SIZE, antialias=True),
            T.ToTensor(),
            T.Normalize(
                mean=img_mean,
                std=img_std,
            ),
        ],
    )

    train_trnsf = T.Compose(
        [
            T.RandomResizedCrop(
                size=INPUT_SIZE,
                scale=(0.6, 1.5),
                ratio=(0.75, 1.33),
                antialias=True,
            ),
            T.RandomVerticalFlip(0.5),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(config["augmentation_settings"]["rotation"]),
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=config["augmentation_settings"]["brigntess_rage"],
                        contrast=config["augmentation_settings"]["contrast_rage"],
                        saturation=config["augmentation_settings"]["saturation"],
                        hue=config["augmentation_settings"]["hue"],
                    )
                ],
                p=0.5,
            ),
            T.TrivialAugmentWide(
                num_magnitude_bins=15, fill=-1
            ),  # this augmentatiton works well for RGB images, but might help with generalization
            T.ToTensor(),
            T.RandomApply([dataset_utilities.AddGaussianNoise(mean=0, std=5)], p=0),
            T.Normalize(
                mean=img_mean,
                std=img_std,
            ),
        ],
    )

    # ## If using a SimCLR model pretrained on CBTN data, we use the same data splits for training the classifier.
    # This is important to avoid data leakage between the pre-training and the classifier.
    # When using a SimCLR pre-trained model on CBTN, the number of repetitions and folds set in the ocnfiguration file are over-written
    # to match the ones of the SimCLR pre-training. The paths of the datasplit .csv files are obtained here and used later in each repetititon

    if all(
        [
            PRETRAINED_TYPE == "SimCLR",
            PRETRAINED_DATASET == "CBTN",
        ]
    ):
        info_from_simclr_pretraining = get_info_from_SimCLR_CBTN_pretraining(config)
    else:
        # ## get heuristics for this dataset type
        (
            heuristic_for_file_discovery,
            heuristic_for_subject_ID_extraction,
            heuristic_for_class_extraction,
            heuristic_for_rlp_extraction,
            heuristic_for_age_extraction,
        ) = dataset_utilities.get_dataset_heuristics(config["dataset"]["name"])

    # %% RUN REPETITIONS
    # check which repetitions need to be run (this is to account for restarting the model training)
    if all(
        [
            "restart_training_settings" in config.keys(),
            config["restart_training_settings"]["restart_training"],
        ]
    ):
        # get all the repetitions that have at least one fold to be run
        repetitions_to_run = [
            r
            for r, f in config["restart_training_settings"]["restart_from"].items()
            if len(f) != 0
        ]
    else:
        repetitions_to_run = list(range(config["training_settings"]["nbr_repetitions"]))

    for repetition_nbr in repetitions_to_run:
        if all(
            [
                PRETRAINED_TYPE == "SimCLR",
                PRETRAINED_DATASET == "CBTN",
            ]
        ):
            # load the data split used for SimCLR training
            dataset_split_df = pd.read_csv(
                info_from_simclr_pretraining[repetition_nbr + 1]["datasplit_path"]
            )
            # check if the paths point to the local machine, if not convert to the local dataset path
            print(os.path.isdir(os.path.dirname(dataset_split_df["file_path"][0])))
            if not os.path.isdir(os.path.dirname(dataset_split_df["file_path"][0])):
                dataset_split_df[f"file_path"] = dataset_split_df.apply(
                    lambda x: os.path.join(
                        config["dataset"]["dataset_path"],
                        os.path.basename(x[f"file_path"]),
                    ),
                    axis=1,
                )

            # print information about training, validation and testing files since not performing the split but loading from file
            dataset_utilities.print_summary_from_dataset_split(dataset_split_df)
        else:
            dataset_split_df = dataset_utilities._get_split(
                config,
                heuristic_for_file_discovery=heuristic_for_file_discovery,
                heuristic_for_subject_ID_extraction=heuristic_for_subject_ID_extraction,
                heuristic_for_class_extraction=heuristic_for_class_extraction,
                heuristic_for_relative_position_extraction=heuristic_for_rlp_extraction,
                heuristic_for_age_extraction=heuristic_for_age_extraction,
                repetition_number=repetition_nbr,
                randomize_subject_labels=config["dataloader_settings"][
                    "randomize_subject_labels"
                ],
                randomize_slice_labels=config["dataloader_settings"][
                    "randomize_slice_labels"
                ],
                select_slices_in_range=config["dataloader_settings"][
                    "select_slices_in_range"
                ],
            )

        # normalize age is requested
        if all(
            [
                config["dataloader_settings"]["use_age"],
                config["dataloader_settings"]["normalize_age"],
            ]
        ):
            dataset_split_df["age_normalized"] = [None] * len(dataset_split_df)
            # do this for each fold separately (only use the training data for the computation of the mean and std)
            for fold in range(config["training_settings"]["nbr_inner_cv_folds"]):
                # get values for this fold
                age_in_days_array = np.array(
                    dataset_split_df.loc[
                        dataset_split_df[f"fold_{fold+1}"] == "training"
                    ]["age_in_days"]
                )
                index_df_rows = dataset_split_df.index[
                    dataset_split_df[f"fold_{fold+1}"] == "training"
                ].tolist()

                # normalize age in days using mean and std in the [0.5, 99.5] percentile
                age_mean = np.mean(
                    age_in_days_array[
                        np.logical_and(
                            age_in_days_array > np.percentile(age_in_days_array, 0.5),
                            age_in_days_array <= np.percentile(age_in_days_array, 99.5),
                        )
                    ]
                )
                age_std = np.std(
                    age_in_days_array[
                        np.logical_and(
                            age_in_days_array > np.percentile(age_in_days_array, 0.5),
                            age_in_days_array <= np.percentile(age_in_days_array, 99.5),
                        )
                    ]
                )

                normalized_age = (age_in_days_array - age_mean) / age_std

                # save values
                dataset_split_df.loc[index_df_rows, "age_normalized"] = (
                    normalized_age.tolist()
                )
                # save nromalization values in the confoguration file
                config["dataloader_settings"]["age_mean"] = age_mean.tolist()
                config["dataloader_settings"]["age_std"] = age_std.tolist()

                # save also the normalized test ages
                index_df_rows = dataset_split_df.index[
                    dataset_split_df[f"fold_{fold+1}"] == "test"
                ].tolist()
                test_ages_in_days = np.array(
                    dataset_split_df.loc[dataset_split_df[f"fold_{fold+1}"] == "test"][
                        "age_in_days"
                    ]
                )
                test_ages_in_days_normalized = (test_ages_in_days - age_mean) / age_std
                dataset_split_df.loc[index_df_rows, "age_normalized"] = (
                    test_ages_in_days_normalized.tolist()
                )
        else:
            dataset_split_df["age_normalized"] = [None] * len(dataset_split_df)

        # %% RUN TRAINING FOR THE DIFFERENT FOLDS
        # check which repetitions need to be run (this is to account for restarting the model training)
        if all(
            [
                "restart_training_settings" in config.keys(),
                config["restart_training_settings"]["restart_training"],
            ]
        ):
            # get all the repetitions that have at least one fold to be run
            folds_to_run = config["restart_training_settings"]["restart_from"][
                repetition_nbr
            ]
        else:
            folds_to_run = list(
                range(config["training_settings"]["nbr_inner_cv_folds"])
            )
        for fold in folds_to_run:
            logger.info(
                f'Working on fold {fold+1}/{config["training_settings"]["nbr_inner_cv_folds"]} of repetition {repetition_nbr+1}/{config["training_settings"]["nbr_repetitions"]}'
            )

            # ## get training validation and testing files for this repetition and fold
            samples_for_training = dataset_split_df.loc[
                dataset_split_df[f"fold_{fold+1}"] == "training"
            ].reset_index()

            samples_for_validation = dataset_split_df.loc[
                dataset_split_df[f"fold_{fold+1}"] == "validation"
            ].reset_index()

            if config["training_settings"]["running_for_final_testing"]:
                # combine the validation and the trainig set so that the model can learn from a largerd sample size.
                # NOTE. This should only be done after all the experimetns are performed and one is ready for the running the model one last time.
                logging.info(
                    'ATTENTION!!!\nRunning training script in "final run mode", where the training and the validation sets are combined.\nThis is to allow the model train on all the non test samples.\nThis should only be the case if you are ready for the last run befor getting the test results.'
                )
                # concatenate training and validation samples
                samples_for_training = pd.concat(
                    [samples_for_training, samples_for_validation]
                )

            samples_for_testing = dataset_split_df.loc[
                dataset_split_df[f"fold_{fold+1}"] == "test"
            ].reset_index()

            # build batch sampler
            training_batch_sampler = None
            if (
                "use_one_slice_per_subject_within_epoch"
                in config["dataloader_settings"].keys()
            ):
                if config["dataloader_settings"][
                    "use_one_slice_per_subject_within_epoch"
                ]:
                    # build batch sampler
                    training_batch_sampler = (
                        dataset_utilities.OneSlicePerPatientBatchSampler(
                            df_dataset=samples_for_training,
                            nbr_batches_per_epoch=int(
                                len(samples_for_training)
                                / config["dataloader_settings"]["batch_size"]
                            ),
                            nbr_samples_per_batch=config["dataloader_settings"][
                                "batch_size"
                            ],
                        )
                    )
                    logger.info(
                        f"Using custom batch sampler (one slice per subject in each epoch.)"
                    )

            training_dataloader = dataset_utilities.CustomDataset(
                train_sample_paths=list(samples_for_training["file_path"]),
                validation_sample_paths=list(samples_for_validation["file_path"]),
                test_sample_paths=list(samples_for_testing["file_path"]),
                training_targets=list(samples_for_training["target"]),
                validation_targets=list(samples_for_validation["target"]),
                test_targets=list(samples_for_testing["target"]),
                return_age=config["dataloader_settings"]["use_age"],
                train_sample_age=(
                    list(samples_for_training["age_normalized"])
                    if config["dataloader_settings"]["normalize_age"]
                    else list(samples_for_training["age_in_days"])
                ),
                validation_sample_age=(
                    list(samples_for_validation["age_normalized"])
                    if config["dataloader_settings"]["normalize_age"]
                    else list(samples_for_validation["age_in_days"])
                ),
                test_sample_age=(
                    list(samples_for_testing["age_normalized"])
                    if config["dataloader_settings"]["normalize_age"]
                    else list(samples_for_testing["age_in_days"])
                ),
                batch_size=config["dataloader_settings"]["batch_size"],
                num_workers=config["dataloader_settings"]["nbr_workers"],
                preprocess=preprocess,
                transforms=(
                    train_trnsf
                    if config["dataloader_settings"]["augmentation"]
                    else preprocess
                ),
                training_batch_sampler=training_batch_sampler,
            )

            # print relation between classes and one hot encodings
            target_class_to_one_hot_mapping = (
                training_dataloader.return_class_to_onehot_encoding_dict()
            )
            logger.info(f"{target_class_to_one_hot_mapping}")

            # get class weights
            if config["training_settings"]["use_class_weights"]:
                CLASS_WEIGHTS, CLASS_ORDER = training_dataloader.get_class_weights()
            else:
                CLASS_WEIGHTS = [
                    1
                    for i in range(
                        len(config["dataloader_settings"]["classes_of_interest"])
                    )
                ]
                CLASS_ORDER = list(
                    dict.fromkeys(config["dataloader_settings"]["classes_of_interest"])
                )
            logger.info(f"Class weights: {CLASS_WEIGHTS} for classes {CLASS_ORDER}")
            logger.info("Done!")

            # %% PLOT HISTOLGRAMS of training validation and training is required
            if config["logging_settings"]["save_training_validation_test_hist"]:
                dataset_utilities._plot_histogram_from_dataloader(
                    training_dataloader, config
                )

            # %% TRAIN MODEL
            save_name = f"{MODEL}_pretrained_{PRETRAINED}_{PRETRAINED_TYPE}_dataset_{PRETRAINED_DATASET}_frozen_{FROZE_WEIGHTS}_{PERCENTAGE_FROZEN}_LR_{LEARNING_RATE}_BATCH_{BATCH_SIZE}_AUGMENTATION_{config['dataloader_settings']['augmentation']}_OPTIM_{config['training_settings']['optimizer']}_SCHEDULER_{config['training_settings']['scheduler']}_MLPNODES_{config['model_settings']['mlp_nodes'][0] if len(config['model_settings']['mlp_nodes'])!=0 else 0}_useAge_{config['dataloader_settings']['use_age'] if 'use_age' in config['dataloader_settings'].keys() else False}_{config['logging_settings']['start_time']}"

            save_path = os.path.join(
                config["working_dir"],
                "trained_model_archive",
                f"{config['dataset']['modality']}_TESTs_{config['logging_settings']['start_day']}",
                save_name,
                f"REPETITION_{repetition_nbr+1}",
            )
            Path(save_path).mkdir(exist_ok=True, parents=True)
            # save data split
            dataset_split_df.to_csv(
                os.path.join(save_path, "data_split_information.csv"),
                index=False,
            )

            # build the model based on specifications
            if PRETRAINED_TYPE == "SimCLR":
                if PRETRAINED_DATASET == "CBTN":
                    SimCLR_model_path = info_from_simclr_pretraining[
                        repetition_nbr + 1
                    ]["per_fold_model_paths"][fold + 1]
                else:
                    SimCLR_model_path = config["model_settings"][
                        "SimCLR_prettrained_model_setitngs"
                    ]["model_path"]
            else:
                SimCLR_model_path = None

            model = model_bucket_CBTN_v1.LitModelWrapper(
                version=MODEL.lower(),
                nbr_classes=len(pd.unique(dataset_split_df["target"])),
                pretrained=PRETRAINED,
                freeze_percentage=PERCENTAGE_FROZEN,
                class_weights=CLASS_WEIGHTS,
                learning_rate=config["training_settings"]["learning_rate"],
                optimizer=config["training_settings"]["optimizer"],
                scheduler=config["training_settings"]["scheduler"],
                use_look_ahead_wrapper=config["training_settings"][
                    "use_look_ahead_wrapper"
                ],
                image_mean=img_mean,
                image_std=img_std,
                mpl_nodes=list(config["model_settings"]["mlp_nodes"]),
                use_SimCLR_pretrained_model=PRETRAINED_TYPE == "SimCLR",
                SimCLR_model_path=SimCLR_model_path,
                labels_as_string=list(target_class_to_one_hot_mapping.keys()),
                use_age=(
                    config["dataloader_settings"]["use_age"]
                    if "use_age" in config["dataloader_settings"].keys()
                    else False
                ),
                age_encoder_mlp_nodes=(
                    config["model_settings"]["age_encoder_mlp_nodes"]
                    if "age_encoder_mlp_nodes" in config["model_settings"].keys()
                    else None
                ),
            ).to(device)

            # save model architecture to file
            old_stdout = sys.stdout
            try:
                if "vit" not in config["model_settings"]["model_version"].lower():
                    logger.info("Saving model architecture to file...")
                    MODEL_SUMMARY_LOG_FILE = open(
                        os.path.join(save_path, "model_architecture.log"), "w"
                    )
                    sys.stdout = MODEL_SUMMARY_LOG_FILE
                    if config["dataloader_settings"]["use_age"]:
                        aus_input = ((3, INPUT_SIZE[0], INPUT_SIZE[1]), (1))
                    else:
                        aus_input = (
                            3,
                            INPUT_SIZE[0],
                            INPUT_SIZE[1],
                        )
                    summary(
                        model,
                        aus_input,
                    )
                    MODEL_SUMMARY_LOG_FILE.close()
                    logger.info("Done!")
            except:
                print(
                    "Failed to save model summary to file using torchsummary. Just printing."
                )
                print(model)
                logger.info("Done!")
            sys.stdout = old_stdout

            trainer = pl.Trainer(
                accelerator="gpu",
                max_epochs=config["training_settings"]["epochs"],
                callbacks=[
                    EarlyStopping(
                        monitor="ptl/valid_classification_loss",
                        mode="min",
                        patience=config["training_settings"]["patience"],
                    ),
                    LearningRateMonitor("epoch"),
                    ModelCheckpoint(
                        dirpath=os.path.join(save_path, f"TB_fold_{fold+1}"),
                        filename=f"CKPT_fold_{fold+1}",
                        save_last=True,
                        mode="max",
                    ),
                ],
                logger=TensorBoardLogger(
                    save_path,
                    name=f"TB_fold_{fold+1}",
                ),
                log_every_n_steps=2,
            )
            trainer.logger._default_hp_metric = None

            # save configuration file just before training
            OmegaConf.save(config, os.path.join(save_path, "hydra_config.yaml"))

            trainer.fit(model, training_dataloader)

            # save last model
            logger.info("Saving last model...")
            torch.save(model, os.path.join(save_path, f"TB_fold_{fold+1}", "last.pt"))

            # %% RUN TESTING IF REQUESTED. This should be done after all the trials are performed to not be biased by the evaluation on the test
            if config["training_settings"]["run_testing"]:
                logger.info("Running testing...")
                # run testing
                # here we shoudl save the metrics in a csv file accessible by the _obrain_test_performance function (TODO)
                logger("Done!")


# %%
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    config = dict(config)
    config = set_up(config)

    print(config)
    run_training(config)


if __name__ == "__main__":
    main()
