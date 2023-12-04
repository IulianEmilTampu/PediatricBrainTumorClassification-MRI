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

"""
Meeting Anders 25/10/2023

- repeat results using the custom model.
- Use ADC data
- Use pyRadiomics to check the data disctibution and features
- Try to feed only the tumor region
- RadImageNet
-  
"""

# local imports
import model_bucket_CBTN_v1
import dataset_utilities


def set_up(config: dict):
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
    logging.getLogger().addHandler(console)

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

    # ---------------------
    # SEED EVERYTHING
    # ---------------------
    pl.seed_everything(
        config["training_settings"]["random_state"]
    )  # To be reproducable


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
            info_from_simclr_pretraining[rep_nbr]["per_fold_model_paths"][
                fold_nbr
            ] = os.path.join(fold, "last.pt")

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
    for repetition_nbr in range(config["training_settings"]["nbr_repetitions"]):
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
                dataset_split_df.loc[
                    index_df_rows, "age_normalized"
                ] = normalized_age.tolist()
        else:
            dataset_split_df["age_normalized"] = [None] * len(dataset_split_df)

        # %% RUN TRAINING FOR THE DIFFERENT FOLDS
        for fold in range(config["training_settings"]["nbr_inner_cv_folds"]):
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
                train_sample_age=list(samples_for_training["age_normalized"])
                if config["dataloader_settings"]["normalize_age"]
                else list(samples_for_training["age_in_days"]),
                validation_sample_age=list(samples_for_validation["age_normalized"])
                if config["dataloader_settings"]["normalize_age"]
                else list(samples_for_validation["age_in_days"]),
                test_sample_age=list(samples_for_testing["age_normalized"])
                if config["dataloader_settings"]["normalize_age"]
                else list(samples_for_testing["age_in_days"]),
                batch_size=config["dataloader_settings"]["batch_size"],
                num_workers=config["dataloader_settings"]["nbr_workers"],
                preprocess=preprocess,
                transforms=train_trnsf
                if config["dataloader_settings"]["augmentation"]
                else preprocess,
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
                # f"TESTs_{datetime.now().strftime('%Y%m%d')}",
                f"TESTs_{config['logging_settings']['start_day']}",
                save_name,
                # "TEST",
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
                use_age=config["dataloader_settings"]["use_age"]
                if "use_age" in config["dataloader_settings"].keys()
                else False,
                age_encoder_mlp_nodes=config["model_settings"]["age_encoder_mlp_nodes"]
                if "age_encoder_mlp_nodes" in config["model_settings"].keys()
                else None,
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
    set_up(config)
    run_training(config)

    if config["training_settings"]["run_testing"]:
        print("Gathering testing performance metrics")


if __name__ == "__main__":
    main()
