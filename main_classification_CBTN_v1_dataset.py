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
Partially freezing is dependent on model architecture.

ResNet50 -> it has 9 children in the model architecture. 
- Childrens 0 to 3 (conv, BatchNorm, ReLU, MaxPool) bring the input from 3 channels to 64 channels
- Childrens 4 (64 -> 256, 10 convs), 5 (256 -> 512, 10 convs), 6 (512 -> 1024, 19 convs) and 7 (1024 -> 2048, 10 convs) are Sequentials making up most of the model.
- Children 8 and 9 are the average pool and the classification layer (children 9 is replaced based on the classification task)

This is the definition of what is frozen when setting PERCENTAGE_FROZEN
- PERCENTAGE_FROZEN = 1: all the model is frozen
- PERCENTAGE_FROZEN = 0.80: children 0 to 6 (7 is training)
- PERCENTAGE_FROZEN = 0.40: children 0 to 5 (7, 6 are training)
- PERCENTAGE_FROZEN = 0.20.: children 0 to 4 (7, 6, 5 are training)
- PERCENTAGE_FROZEN = 0.05: children 0 to 3 (7, 6, 5, 4 are training)
- PERCENTAGE_FROZEN = 0.0: all the model if training


ResNet18 -> it has 9 children in the model architecture. 
- Childrens 0 to 3 (conv, BatchNorm, ReLU, MaxPool) bring the input from 3 channels to 64 channels
- Childrens 4 (64 -> 64, 4 convs), 5 (64 -> 128, 5 convs), 6 (128 -> 256, 5 convs) and 7 (256 -> 512, 5 convs) are Sequentials making up most of the model.
- Children 8 and 9 are the average pool and the classification layer (children 9 is replaced based on the classification task)

This is the definition of what is frozen when setting PERCENTAGE_FROZEN
- PERCENTAGE_FROZEN = 1: all the model is frozen
- PERCENTAGE_FROZEN = 0.80: children 0 to 6 (7 is training)
- PERCENTAGE_FROZEN = 0.60: children 0 to 5 (7, 6 are training)
- PERCENTAGE_FROZEN = 0.40.: children 0 to 4 (7, 6, 5 are training)
- PERCENTAGE_FROZEN = 0.05: children 0 to 3 (7, 6, 5, 4 are training)
- PERCENTAGE_FROZEN = 0.0: all the model if training

ResNet9 -> it has 6 children in the model architecture. (https://github.com/Moddy2024/ResNet-9/tree/main)
- Children 0 (conv, BatchNorm, ReLU) brings the input from 3 channels to 64 channels
- Children 1 (conv, BatchNorm, ReLU, MaxPool) 64 -> 128
- Children 2 (conv, BatchNorm, ReLU, conv, BatchNorm, ReLU) 128 -> 128
- Children 3 (conv, BatchNorm, ReLU, MaxPool) 128 -> 256
- Children 4 (conv, BatchNorm, ReLU, MaxPool) 256 -> 512
- Children 5 (conv, BatchNorm, ReLU, conv, BatchNorm, ReLU) 512 -> 512
- Children 6 (GlobalAveragePool, Flatten, Dropout, Dense) 512 -> classes (this is replaced by the new classification head)

This is the definition of what is frozen when setting PERCENTAGE_FROZEN
- PERCENTAGE_FROZEN = 1: all the model is frozen
- PERCENTAGE_FROZEN = 0.80: children 0 to 4 (5 is training)
- PERCENTAGE_FROZEN = 0.60: children 0 to 3 (5, 4 are training)
- PERCENTAGE_FROZEN = 0.40.: children 0 to 2 (5, 4, 3 are training)
- PERCENTAGE_FROZEN = 0.20.: children 0 to 1 (5, 4, 3, 2 are training)
- PERCENTAGE_FROZEN = 0.05: children 0 (5, 4, 3, 2, 1 are training)
- PERCENTAGE_FROZEN = 0.0: all the model if training

ViT vit_b_N - > it has 3 children of which the last is the classification head and child 1 is the encoder.
The encoder has 3 children of which the second is a Sequential with 12 EncoderBlocks (this can be accessed using len(net.encoder.layers))
Here the fraction of layers to be frozen depends on the length of the Sequential layer in the model encoder. 
So, if there are 12 encoding blocks, a fraction of 0.7 will freeze int(12 * 0.7) encoding layers.
"""

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

    # save the starting time
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


def run_training(config: dict) -> None:
    logger = logging.getLogger("main.run_training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #  load HP from configyaml file
    PRETRAINED = config["model_settings"]["pre_trained"]
    FROZE_WEIGHTS = config["model_settings"]["freeze_weights"]
    PERCENTAGE_FROZEN = config["model_settings"]["percentage_freeze_weights"]
    BATCH_SIZE = config["dataloader_settings"]["batch_size"]
    LEARNING_RATE = config["training_settings"]["learning_rate"]
    MODEL = config["model_settings"]["model_version"]
    INPUT_SIZE = list(config["dataloader_settings"]["input_size"])

    # %% DATA GENERATOR
    # ## define mean and std for normalization
    ImageNet_ResNet_mean = np.array([0.4451, 0.4262, 0.3959])
    ImageNet_ResNet_std = np.array([0.2411, 0.2403, 0.2466])

    ImageNet_VGG_mean = np.array([0.485, 0.456, 0.406])
    ImageNet_VGG_std = np.array([0.229, 0.224, 0.225])

    MR_mean = np.array([0.5, 0.5, 0.5])
    MR_std = np.array([0.5, 0.5, 0.5])

    # select the mean and std for the normalization
    if all([PRETRAINED, not FROZE_WEIGHTS]):
        # use ImageNet stats since the weights were trained using those stats
        if "resnet" in config["model_settings"]["model_version"].lower():
            img_mean = ImageNet_ResNet_mean
            img_std = ImageNet_ResNet_std
        elif "vgg" in config["model_settings"]["model_version"].lower():
            img_mean = ImageNet_VGG_mean
            img_std = ImageNet_VGG_std
    else:
        # use MR stats since the encoder weights are trained
        img_mean = MR_mean
        img_std = MR_std

    # img_mean = MR_mean
    # img_std = MR_std

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

    # ## get heuristics for this dataset type
    (
        heuristic_for_file_discovery,
        heuristic_for_subject_ID_extraction,
        heuristic_for_class_extraction,
        heuristic_for_rlp_extraction,
    ) = dataset_utilities.get_dataset_heuristics(config["dataset"]["name"])

    # %% RUN REPETITIONS
    for repetition_nbr in range(config["training_settings"]["nbr_repetitions"]):
        dataset_split_df = dataset_utilities._get_split(
            config,
            heuristic_for_file_discovery=heuristic_for_file_discovery,
            heuristic_for_subject_ID_extraction=heuristic_for_subject_ID_extraction,
            heuristic_for_class_extraction=heuristic_for_class_extraction,
            heuristic_for_relative_position_extraction=heuristic_for_rlp_extraction,
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
        # %% RUN TRAINING FOR THE DIFFERENT FOLDS
        for fold in range(config["training_settings"]["nbr_inner_cv_folds"]):
            logger.info(
                f'Working on fold {fold+1}/({config["training_settings"]["nbr_inner_cv_folds"]}) of repetition {repetition_nbr+1}/{config["training_settings"]["nbr_repetitions"]}'
            )

            # ## get training validation and testing files for this repetition and fold
            files_for_training = list(
                dataset_split_df.loc[dataset_split_df[f"fold_{fold+1}"] == "training"][
                    "file_path"
                ]
            )
            training_targets = list(
                dataset_split_df.loc[dataset_split_df[f"fold_{fold+1}"] == "training"][
                    "target"
                ]
            )
            files_for_validation = list(
                dataset_split_df.loc[
                    dataset_split_df[f"fold_{fold+1}"] == "validation"
                ]["file_path"]
            )
            validation_targets = list(
                dataset_split_df.loc[
                    dataset_split_df[f"fold_{fold+1}"] == "validation"
                ]["target"]
            )

            files_for_testing = list(
                dataset_split_df.loc[dataset_split_df[f"fold_{fold+1}"] == "test"][
                    "file_path"
                ]
            )
            test_targets = list(
                dataset_split_df.loc[dataset_split_df[f"fold_{fold+1}"] == "test"][
                    "target"
                ]
            )

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
                            df_dataset=dataset_split_df.loc[
                                dataset_split_df[f"fold_{fold+1}"] == "training"
                            ].reset_index(),
                            nbr_batches_per_epoch=int(
                                len(files_for_training)
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
                train_sample_paths=files_for_training,
                validation_sample_paths=files_for_validation,
                test_sample_paths=files_for_testing,
                training_targets=training_targets,
                validation_targets=validation_targets,
                test_targets=test_targets,
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
            save_name = f"{MODEL}_pretrained_{PRETRAINED}_frozen_{FROZE_WEIGHTS}_{PERCENTAGE_FROZEN}_LR_{LEARNING_RATE}_BATCH_{BATCH_SIZE}_AUGMENTATION_{config['dataloader_settings']['augmentation']}_OPTIM_{config['training_settings']['optimizer']}_SCHEDULER_{config['training_settings']['scheduler']}_MLPNODES_{config['model_settings']['mlp_nodes'][0] if len(config['model_settings']['mlp_nodes'])!=0 else 0}_{config['logging_settings']['start_time']}"
            save_path = os.path.join(
                config["working_dir"],
                "trained_model_archive",
                # f"TESTs_{datetime.now().strftime('%Y%m%d')}_2DSDM4_exp",
                f"TESTs_{datetime.now().strftime('%Y%m%d')}",
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
                use_SimCLR_pretrained_model=config["model_settings"]["use_SimCLR_model"]
                if "use_SimCLR_model" in config["model_settings"].keys()
                else False,
                SimCLR_model_path=config["model_settings"]["SimCLR_model_setitngs"][
                    "model_path"
                ]
                if "SimCLR_model_setitngs" in config["model_settings"].keys()
                else None,
                labels_as_string=list(target_class_to_one_hot_mapping.keys()),
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
                    summary(
                        model,
                        (
                            3,
                            INPUT_SIZE[0],
                            INPUT_SIZE[1],
                        ),
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
