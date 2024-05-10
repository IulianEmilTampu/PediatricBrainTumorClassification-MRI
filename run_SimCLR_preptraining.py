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


# local imports
import core_utils.model_utilities
import core_utils.dataset_utilities


def set_up(config: dict):
    # save the starting time for the SimCLR training
    config["logging_settings"]["SimCLR_start_time"] = datetime.now().strftime("t%H%M%S")

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

    # ---------------------
    # Set GPU to be used
    # ---------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["resources"]["gpu_nbr"])

    # ---------------------
    # SEED EVERYTHING
    # ---------------------
    pl.seed_everything(
        config["SimCLR"]["training_settings"]["random_state"]
    )  # To be reproducable


def run_SimCLR_training(config: dict) -> None:
    logger = logging.getLogger("main.run_training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get out hp from config.yaml dictionary (this is just for convenience)
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

    # ImageNet_VGG_mean = np.array([0.485, 0.456, 0.406])
    # ImageNet_VGG_std = np.array([0.229, 0.224, 0.225])

    MR_mean = np.array([0.5, 0.5, 0.5])
    MR_std = np.array([0.5, 0.5, 0.5])

    # select the mean and std for the normalization
    if all([PRETRAINED, not FROZE_WEIGHTS]):
        # use ImageNet stats since the weights were trained using those stats
        if "resnet" in config["model_settings"]["model_version"].lower():
            img_mean = ImageNet_ResNet_mean
            img_std = ImageNet_ResNet_std
        # elif 'vgg' in config['SimCLR']["model_settings"]["model_version"].lower():
        #     img_mean = ImageNet_VGG_mean
        #     img_std = ImageNet_VGG_std
    else:
        # use MR stats since the encoder weights are trained
        img_mean = MR_mean
        img_std = MR_std

    config["dataloader_settings"]["img_mean"] = img_mean.tolist()
    config["dataloader_settings"]["img_std"] = img_std.tolist()

    # ## define contrastive transforms
    contrastive_transforms = T.Compose(
        [
            T.RandomResizedCrop(
                size=INPUT_SIZE,
                scale=(0.6, 1.5),
                ratio=(0.75, 1.33),
                antialias=True,
            ),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(config["augmentation_settings"]["rotation"]),
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=config["augmentation_settings"]["brigntess_rage"],
                        contrast=config["augmentation_settings"]["contrast_rage"],
                        saturation=config["augmentation_settings"]["saturation"],
                        hue=config["augmentation_settings"]["hue"],
                    ),
                    T.GaussianBlur(kernel_size=9),
                ],
                p=0.8,
            ),
            T.RandomApply(
                [T.TrivialAugmentWide(num_magnitude_bins=15, fill=-1)],
                p=1 if config["augmentation_settings"]["use_trivialAugWide"] else 0,
            ),
            T.ToTensor(),
            T.Normalize(
                mean=img_mean,
                std=img_std,
            ),
        ],
    )

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

            if config["training_settings"]["running_for_final_testing"]:
                # combine the validation and the trainig set so that the model can learn from a largerd sample size.
                # NOTE. This should only be done after all the experimetns are performed and one is ready for the running the model one last time.
                logging.info(
                    'ATTENTION!!!\nRunning training script in "final run mode", where the training and the validation sets are combined.\nThis is to allow the model train on all the non test samples.\nThis should only be the case if you are ready for the last run befor getting the test results.'
                )
                # concatenate training and validation samples
                files_for_training.extend(files_for_validation)
                training_targets.extend(validation_targets)

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
            if "use_slices_as_views" in config["dataloader_settings"].keys():
                if config["dataloader_settings"]["use_slices_as_views"]:
                    # build batch sampler
                    training_batch_sampler = (
                        dataset_utilities.SimCLRBatchSamplerTwoSubjectSlicesAsView(
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
                        f"Using custom batch sampler (views as different slices from the same subject)."
                    )

            training_dataloader = dataset_utilities.CustomDataset(
                train_sample_paths=files_for_training,
                validation_sample_paths=files_for_validation,
                test_sample_paths=files_for_testing,
                batch_size=config["dataloader_settings"]["batch_size"],
                num_workers=config["dataloader_settings"]["nbr_workers"],
                preprocess=dataset_utilities.ContrastiveTransformations(
                    contrastive_transforms, n_views=2
                ),
                transforms=dataset_utilities.ContrastiveTransformations(
                    contrastive_transforms, n_views=2
                ),
                training_batch_sampler=training_batch_sampler,
            )
            # get class weights
            logger.info("Done!")

            # %% TRAIN MODEL"
            save_name = f"SimCLR_{MODEL}_pretrained_{PRETRAINED}_frozen_{FROZE_WEIGHTS}_{PERCENTAGE_FROZEN}_LR_{LEARNING_RATE}_BATCH_{BATCH_SIZE}_TrivialAug_{config['augmentation_settings']['use_trivialAugWide']}_{config['logging_settings']['SimCLR_start_time']}"
            save_path = os.path.join(
                config["working_dir"],
                "trained_model_archive",
                # f"TESTs_{datetime.now().strftime('%Y%m%d')}_2DSDM4_exp",
                f"{config['dataset']['modality']}_SimCLR_TESTs_{datetime.now().strftime('%Y%m%d')}",
                save_name,
                f"REPETITION_{repetition_nbr+1}",
            )
            Path(save_path).mkdir(exist_ok=True, parents=True)
            Path(os.path.join(save_path, f"TB_fold_{fold+1} "))

            # save data split
            dataset_split_df.to_csv(
                os.path.join(save_path, "data_split_information.csv"),
                index=False,
            )

            model = model_bucket_CBTN_v1.SimCLRModelWrapper(
                version=MODEL.lower(),
                pretrained=PRETRAINED,
                freeze_percentage=PERCENTAGE_FROZEN,
                lr=config["training_settings"]["learning_rate"],
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
                    print(model)
                    MODEL_SUMMARY_LOG_FILE.close()
                    logger.info("Done!")
            except:
                print("Failed to save model summary to file.")
            sys.stdout = old_stdout

            trainer = pl.Trainer(
                accelerator="gpu",
                max_epochs=config["training_settings"]["epochs"],
                callbacks=[
                    EarlyStopping(
                        monitor="ptl/val_acc_top5",
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
                log_every_n_steps=5,
            )
            trainer.logger._default_hp_metric = None

            # save configuration file just before training
            OmegaConf.save(config, os.path.join(save_path, "hydra_config.yaml"))

            trainer.fit(model, training_dataloader)

            # save last model
            logger.info("Saving last model...")
            try:
                torch.save(
                    model, os.path.join(save_path, f"TB_fold_{fold+1}", "last.pt")
                )
            except:
                logger.info(
                    "Failing to save model using torch.save. Loading the last checkpoint and saving."
                )
                # delete the model
                del model
                # re initialize from checkpoint
                model = model_bucket_CBTN_v1.SimCLRModelWrapper.load_from_checkpoint(
                    os.path.join(save_path, f"TB_fold_{fold+1}", "last.ckpt"),
                    version=MODEL.lower(),
                    pretrained=PRETRAINED,
                    freeze_percentage=PERCENTAGE_FROZEN,
                    lr=config["training_settings"]["learning_rate"],
                )
                # save model
                try:
                    torch.save(
                        model, os.path.join(save_path, f"TB_fold_{fold+1}", "last.pt")
                    )
                except:
                    logger.info(
                        "Failing to save model using torch.save even after reinitialization from checkpoint. Use checkpoint to load the model in the future."
                    )


# %%
@hydra.main(version_base=None, config_path="conf", config_name="SimCLR_config")
def main(config: DictConfig):
    config = dict(config)
    set_up(config)

    # take out configurations for the SimCLR training and put all into a dictionary
    SimCLR_conf = dict(config["SimCLR"])
    SimCLR_conf["working_dir"] = config["working_dir"]
    SimCLR_conf["dataset"] = config["dataset"]
    SimCLR_conf["debugging_settings"] = config["debugging_settings"]
    SimCLR_conf["logging_settings"] = config["logging_settings"]

    # run SimCLR training
    run_SimCLR_training(SimCLR_conf)

    # if config["training_settings"]["run_testing"]:
    #     print("Gathering testing performance metrics")


if __name__ == "__main__":
    main()
