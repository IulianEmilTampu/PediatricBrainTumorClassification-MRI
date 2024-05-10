# %%
"""
https://github.com/jacobgil/pytorch-grad-cam/

Script that runs GradCAM on a selection of specific subjects. It runs the models on all the repetitions and folds (quite time consuming). 
Here we save the images for one patient in one image (ordered by the relative position).
"""
import os
import glob
import pandas as pd
import random
import numpy as np
import tqdm
import PIL
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib.pyplot as plt
import cv2
import pathlib
import importlib
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
from datetime import datetime
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from omegaconf import DictConfig, OmegaConf

import torchvision
import torchvision.transforms as transforms

from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# local imports
import model_bucket_CBTN_v1
import dataset_utilities

os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


# %% PATHS
# settings
USE_AUGMENTATION = False
MR_MODALITY_TO_PLOT = "T2"
NBR_IMGS_PER_ROW = 6
PATH_TO_LOCAL_DATASET_CONFIGS = (
    "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/conf/dataset"
)
# collect paths to models from the given folder
PATH_TO_MODELS_TO_INVESTIGATE = "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/count_used_subjects_slices/list_og_model_to_evaluate_GradCAM.txt"

with open(PATH_TO_MODELS_TO_INVESTIGATE, "r") as f:
    PATH_TO_MODELs_CKTP = [line.rstrip("\n") for line in f]

# for debugging
# PATH_TO_MODELs_CKTP = PATH_TO_MODELs_CKTP[0:5]

# path to the .csv file pointing to the specific subjects to plot
PER_MODALITY_specific_subjects = {
    "ADC": "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/count_used_subjects_slices/ADC_subjects_with_all_mr_sequences.csv",
    "T1": "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/count_used_subjects_slices/T1_subjects_with_all_mr_sequences.csv",
    "T2": "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/count_used_subjects_slices/T2_subjects_with_all_mr_sequences.csv",
}

# where to save
SAVE_PATH = os.path.join(
    os.getcwd(),
    "GradCAM_evaluation",
    datetime.now().strftime("%Y%m%d"),
    "subject_specific_plots",
    MR_MODALITY_TO_PLOT,
    f"Augmentation_{USE_AUGMENTATION}",
)
pathlib.Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

# %% LOOP THROUGH ALL THE MODELS
with tqdm.tqdm(
    PATH_TO_MODELs_CKTP,
    position=0,
    desc="Models",
    ncols=100,
    leave=True,
    unit=" model",
    total=len(PATH_TO_MODELs_CKTP),
) as model_tqdm:
    for PATH_TO_MODEL_CKTP in model_tqdm:
        PATH_TO_MODEL_CKTP = pathlib.Path(PATH_TO_MODEL_CKTP)
        REPETITION = PATH_TO_MODEL_CKTP.parts[-3].replace("REPETITION_", "")
        FOLD_NAME = PATH_TO_MODEL_CKTP.parts[-2].replace("TB_", "")
        FOLD = int(FOLD_NAME.split("_")[-1])
        MODEL_VERSION = PATH_TO_MODEL_CKTP.parts[-4].split("_")[0]
        PATH_TO_SPLIT_CSV = os.path.join(
            os.path.dirname(os.path.dirname(PATH_TO_MODEL_CKTP)),
            "data_split_information.csv",
        )
        PATH_TO_YAML_CONFIG_FILE = os.path.join(
            os.path.dirname(os.path.dirname(PATH_TO_MODEL_CKTP)), "hydra_config.yaml"
        )

        # load .yaml file as well
        training_config = OmegaConf.load(PATH_TO_YAML_CONFIG_FILE)
        CLASSES_TO_USE = training_config.dataset.classes_of_interest
        IMG_MEAN_NORM = training_config.dataloader_settings.img_mean
        IMG_STD_NORM = training_config.dataloader_settings.img_std
        USE_AGE = training_config.dataloader_settings.use_age
        USE_NORMALIZED_AGE = training_config.dataloader_settings.normalize_age
        MODEL_NAME = pathlib.Path(PATH_TO_MODEL_CKTP).parts[-4]

        if not USE_AGE:
            # check if the model has been trained on a different machine.
            # This will require changing the path to the files in the dataset_split.csv file
            PATH_TO_DATASET_FROM_TRAINING = training_config.dataset.dataset_path
            # check that this points to the local machine
            flag_model_trained_on_a_different_machine = False
            if not os.path.isdir(PATH_TO_DATASET_FROM_TRAINING):
                flag_model_trained_on_a_different_machine = True
                # need to change
                classes_long_name_to_short = {
                    "ASTROCYTOMA": "ASTR",
                    "EPENDYMOMA": "EP",
                    "MEDULLOBLASTOMA": "MED",
                }

                classes_string = "_".join(
                    [
                        classes_long_name_to_short[c]
                        for c in list(training_config["dataset"]["classes_of_interest"])
                    ]
                )
                path_to_local_yaml_dataset_configuration_file = os.path.join(
                    PATH_TO_LOCAL_DATASET_CONFIGS,
                    "_".join(
                        [
                            training_config.dataset.modality,
                            training_config.dataset.name,
                            classes_string,
                        ]
                    )
                    + ".yaml",
                )
                # load local.yaml and save local dataset path
                local_dataset_config = OmegaConf.load(
                    path_to_local_yaml_dataset_configuration_file
                )

                local_dataset_path = local_dataset_config.dataset_path

            # %% LOAD MODEL and define layer to use
            net = torch.load(PATH_TO_MODEL_CKTP)
            if "vit" in MODEL_VERSION.lower():
                if USE_AGE:
                    net = net.to(device)
                    target_layers = [net.model.model.encoder.layers[-1].ln_1]
                else:
                    net = net.model.model.to(device)
                    target_layers = [net.encoder.layers[-1].ln_1]
                use_reshape_transfor = True
            else:
                if USE_AGE:
                    net = net.to(device)
                    target_layers = [net.model.model.layer4[-1].conv3]
                else:
                    net = net.model.to(device)
                    target_layers = [net.model.layer4[-1].conv3]
                use_reshape_transfor = False

            # %% LOOP THROUGH ALL THE SETS_TO_PLOT
            # load the datasetsplit .csv file and subject specific files
            importlib.reload(dataset_utilities)

            subjects_to_run = pd.read_csv(
                PER_MODALITY_specific_subjects[MR_MODALITY_TO_PLOT]
            )
            dataset_split = pd.read_csv(PATH_TO_SPLIT_CSV)
            try:
                dataset_split = dataset_split.drop(columns=["level_0", "index"])
            except:
                pass

            # filter based on the specific subject files
            dataset_split = dataset_split.loc[
                dataset_split.subject_IDs.isin(pd.unique(subjects_to_run.subject_IDs))
            ]

            # fix dataset paths if the model has been trained on a different pc
            if flag_model_trained_on_a_different_machine:
                dataset_split[f"file_path"] = dataset_split.apply(
                    lambda x: os.path.join(
                        local_dataset_path,
                        os.path.basename(x[f"file_path"]),
                    ),
                    axis=1,
                )

            # define transforms
            train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        224, scale=(0.5, 1.5), ratio=(0.7, 1.33)
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(45),
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                            )
                        ],
                        p=0.5,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        np.array(IMG_MEAN_NORM), np.array(IMG_STD_NORM)
                    ),
                ]
            )

            validation_transform = transforms.Compose(
                [
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        np.array(IMG_MEAN_NORM), np.array(IMG_STD_NORM)
                    ),
                ]
            )
            with tqdm.tqdm(
                ["test"],
                position=1,
                desc="Sets",
                ncols=100,
                leave=False,
                unit="set",
                total=len(pd.unique(dataset_split[FOLD_NAME])),
            ) as set_tqdm:
                for set_to_plot in set_tqdm:
                    # create save folder in the model repetition folder
                    save_path = pathlib.Path(
                        os.path.join(
                            SAVE_PATH,
                            MODEL_NAME,
                            REPETITION,
                            FOLD_NAME,
                            set_to_plot,
                        )
                    )
                    save_path.mkdir(parents=True, exist_ok=True)

                    # create datagenerator for the given set
                    g = torch.Generator()
                    g.manual_seed(training_config.training_settings.random_state)

                    samples = dataset_split.loc[
                        (dataset_split[FOLD_NAME] == set_to_plot)
                        & (dataset_split["target"].isin(CLASSES_TO_USE))
                    ]
                    samples = samples.reset_index()

                    # fix the missing age normalized values for the test set
                    if np.isnan(list(samples["age_normalized"])[0]):
                        print("Fixing normalized age in the testing set...")
                        # retrieve the age in days from the file name
                        test_ages = np.array(samples["age_in_days"])
                        # get the mean and std deviation from the training cases in this fold
                        train_ages = np.array(
                            dataset_split.loc[
                                dataset_split[f"fold_{FOLD}"] == "training"
                            ]["age_in_days"]
                        )
                        age_mean = np.mean(
                            train_ages[
                                np.logical_and(
                                    train_ages > np.percentile(train_ages, 0.5),
                                    train_ages <= np.percentile(train_ages, 99.5),
                                )
                            ]
                        )
                        age_std = np.std(
                            train_ages[
                                np.logical_and(
                                    train_ages > np.percentile(train_ages, 0.5),
                                    train_ages <= np.percentile(train_ages, 99.5),
                                )
                            ]
                        )
                        # normalize and save age values
                        test_ages_normalized = (test_ages - age_mean) / age_std
                        samples.loc[:, "age_normalized"] = test_ages_normalized.tolist()

                    sample_dataloader = DataLoader(
                        dataset_utilities.PNGDatasetFromFolder(
                            list(samples["file_path"]),
                            transform=(
                                train_transform
                                if USE_AUGMENTATION
                                else validation_transform
                            ),
                            labels=dataset_utilities.str_class_to_numeric(
                                list(samples["target"]),
                                str_unique_classes=CLASSES_TO_USE,
                                one_hot=True,
                                as_torch_tensors=True,
                            ),
                            return_age=USE_AGE,
                            ages=(
                                list(samples["age_normalized"])
                                if USE_NORMALIZED_AGE
                                else list(samples["age_in_days"])
                            ),
                        ),
                        batch_size=64,
                        num_workers=15,
                        shuffle=False,
                        worker_init_fn=seed_worker,
                        generator=g,
                    )

                    # ## GET GRAD CAMS FOR ALL THE BATCHES IN THE LOADER
                    label_aggregator = []
                    prediction_aggregator = []
                    batch_img_aggregator = []
                    grayscale_cam_aggregator = []

                    with tqdm.tqdm(
                        sample_dataloader,
                        position=2,
                        desc="Batches",
                        ncols=100,
                        leave=False,
                        unit="batch",
                        total=len(sample_dataloader),
                    ) as batch_tqdm:
                        for batch in batch_tqdm:
                            if USE_AGE:
                                model_input = (batch[0].to(device), batch[2].to(device))
                                images = batch[0]
                                labels = batch[1]
                            else:
                                model_input = batch[0].to(device)
                                images = batch[0]
                                labels = batch[1]
                            labels = torch.argmax(labels, axis=1).numpy()

                            # get model predictions on this batch and print MCC and accuracy
                            with torch.no_grad():
                                pred = net(model_input)
                                y_ = torch.argmax(pred, axis=1).to("cpu").numpy()

                            cam = EigenCAM(
                                model=net,
                                target_layers=target_layers,
                                use_cuda=False,
                                reshape_transform=(
                                    reshape_transform if use_reshape_transfor else None
                                ),
                            )

                            # compute GradCAM for all the images in the batch
                            targets = [ClassifierOutputTarget(c) for c in list(y_)]
                            grayscale_cam = cam(
                                input_tensor=model_input,
                                targets=targets,
                                aug_smooth=False,
                                eigen_smooth=True,
                            )

                            # save
                            batch_img_aggregator.append(
                                model_input[0].to("cpu").numpy()
                                if USE_AGE
                                else model_input.to("cpu").numpy()
                            )
                            grayscale_cam_aggregator.append(grayscale_cam)
                            label_aggregator.append(labels)
                            prediction_aggregator.append(pred.to("cpu").numpy())

                    # make np array
                    batch_img_aggregator = np.vstack(batch_img_aggregator)
                    grayscale_cam_aggregator = np.vstack(grayscale_cam_aggregator)
                    label_aggregator = np.hstack(label_aggregator)
                    prediction_aggregator = np.vstack(prediction_aggregator)

                    # %% PLOT EACH SUBJECT SEPARATELY
                    with tqdm.tqdm(
                        pd.unique(samples.subject_IDs),
                        position=2,
                        desc="Subject plots",
                        ncols=100,
                        leave=False,
                        unit="subject",
                        total=len(pd.unique(samples.subject_IDs)),
                    ) as subject_plot_tqdm:
                        for subject in subject_plot_tqdm:
                            # get the indexes for this subject from the samples dataframe
                            subject_indexes = samples.index[
                                samples.subject_IDs == subject
                            ].tolist()

                            # get rgb images, grayscale cams, label and pred
                            subject_rgb_img = batch_img_aggregator[subject_indexes]
                            subject_grayscale_cam = grayscale_cam_aggregator[
                                subject_indexes
                            ]
                            subject_label = label_aggregator[subject_indexes]
                            subject_preds = prediction_aggregator[subject_indexes]
                            subject_slices_rlp = samples.iloc[subject_indexes][
                                "tumor_relative_position"
                            ].to_list()

                            # define nbr of rows and columns (columns are specified in the settings above)
                            n_cols = NBR_IMGS_PER_ROW
                            n_rows = np.ceil(
                                len(subject_indexes) / NBR_IMGS_PER_ROW
                            ).astype(int)

                            # create figure (for cam and original image)
                            fig_cam = plt.figure(figsize=(n_rows * 5, n_cols * 5))
                            grid_gam = ImageGrid(
                                fig_cam,
                                111,  # similar to subplot(111)
                                nrows_ncols=(
                                    n_rows,
                                    n_cols,
                                ),  # creates 2x2 grid of axes
                                axes_pad=(0.05, 0.3),  # pad between axes in inch.
                            )
                            fig_orig = plt.figure(figsize=(n_rows * 5, n_cols * 5))
                            grid_orig = ImageGrid(
                                fig_orig,
                                111,  # similar to subplot(111)
                                nrows_ncols=(
                                    n_rows,
                                    n_cols,
                                ),  # creates 2x2 grid of axes
                                axes_pad=(0.05, 0.3),  # pad between axes in inch.
                            )

                            # fill in the axis
                            for ax_cam, ax_orig, im in zip(
                                grid_gam, grid_orig, range(len(subject_indexes))
                            ):
                                g_images_rgb = np.transpose(
                                    subject_rgb_img[im], (1, 2, 0)
                                )

                                # un-normalize (undo T.Normalize)
                                g_images_rgb = g_images_rgb * np.array(
                                    IMG_MEAN_NORM
                                ) + np.array(IMG_STD_NORM)

                                # bring in [0,1]
                                g_images_rgb = (g_images_rgb - g_images_rgb.min()) / (
                                    g_images_rgb.max() - g_images_rgb.min()
                                )

                                # overlay g_image_rgb and cam
                                visualization = show_cam_on_image(
                                    g_images_rgb,
                                    subject_grayscale_cam[im],
                                    use_rgb=True,
                                )

                                # rotate images if needed (only if not ADC)
                                if MR_MODALITY_TO_PLOT != "ADC":
                                    visualization = np.rot90(visualization, 1)
                                    g_images_rgb = np.rot90(g_images_rgb, 1)

                                # plot and fix ax_cam
                                ax_cam.imshow(visualization / 255)
                                ax_cam.set_title(
                                    f'rlp: {subject_slices_rlp[im]:0.1f}, GT: {subject_label[im]}, P: {np.argmax(subject_preds[im])}\nP: {[f"{v:0.2f}" for v in subject_preds[im]]}',
                                    fontsize=7,
                                )
                                ax_cam.set_axis_off()

                                # plot and fix ax_orig
                                ax_orig.imshow(g_images_rgb)
                                ax_orig.set_title(
                                    f"rlp: {subject_slices_rlp[im]:0.1f}, GT: {subject_label[im]}",
                                    fontsize=7,
                                )
                                ax_orig.set_axis_off()

                            # save image for this subject
                            fig_cam.savefig(
                                fname=os.path.join(
                                    save_path,
                                    f"{subject}_{set_to_plot}_gradCAM_img.pdf",
                                ),
                                dpi=100,
                                format="pdf",
                                bbox_inches="tight",
                            )
                            fig_orig.savefig(
                                fname=os.path.join(
                                    save_path,
                                    f"{subject}_{set_to_plot}_original_img.pdf",
                                ),
                                dpi=100,
                                format="pdf",
                                bbox_inches="tight",
                            )

                            # fig_cam.savefig(
                            #     fname=os.path.join(
                            #         save_path,
                            #         f"{subject}_{set_to_plot}_gradCAM_img.png",
                            #     ),
                            #     dpi=100,
                            #     format="png",
                            #     bbox_inches="tight",
                            # )

                            # fig_orig.savefig(
                            #     fname=os.path.join(
                            #         save_path,
                            #         f"{subject}_{set_to_plot}_original_img.png",
                            #     ),
                            #     dpi=100,
                            #     format="png",
                            #     bbox_inches="tight",
                            # )
                            plt.close(fig_cam)
                            plt.close(fig_orig)
                    # except:
                    #     continue
