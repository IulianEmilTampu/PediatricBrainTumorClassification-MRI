# %%
"""
https://github.com/jacobgil/pytorch-grad-cam/
"""
import os
import glob
import pandas as pd
import random
import numpy as np
import PIL
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib.pyplot as plt
import cv2
import pathlib
import importlib
from matplotlib import cm
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
import core_utils.model_utilities
import core_utils.dataset_utilities

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
PATH_TO_LOCAL_DATASET_CONFIGS = (
    "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/conf/dataset"
)

# collect paths to models from the given folder
PATH_TO_TRAINED_MODELS = "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/train_model_archive_POST_20231208"

# settings
# SETS_TO_PLOT = ["training", "validation", "test"]
SETS_TO_PLOT = ["test"]
USE_AUGMENTATION = False
MR_MODALITY_TO_PLOT = "T2"
REPETITION = 5
FOLD = 3
PATH_TO_MODELs_CKTP = []

for mr_sequence_folder in glob.glob(os.path.join(PATH_TO_TRAINED_MODELS, "*", "")):
    if MR_MODALITY_TO_PLOT in pathlib.Path(mr_sequence_folder).parts[-1]:
        for m in glob.glob(os.path.join(mr_sequence_folder, "*", "")):
            # for this model build path to the .pt file for the repetition and the fold specified
            ckpt_path = os.path.join(
                m, f"REPETITION_{REPETITION}", f"TB_fold_{FOLD}", "last.pt"
            )
            # check that the file exists
            if not os.path.isfile(ckpt_path):
                print(".pt file not found")
            else:
                # add the model to the list to process.
                # just process ViT models
                if "ViT_b_16" in m:
                    PATH_TO_MODELs_CKTP.append(ckpt_path)

PATH_TO_MODELs_CKTP = [
    "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/train_model_archive_POST_20231208/ADC_TESTs_20240128/ViT_b_16_pretrained_True_SimCLR_dataset_CBTN_frozen_True_0.5_LR_1e-05_BATCH_128_AUGMENTATION_True_OPTIM_adam_SCHEDULER_exponential_MLPNODES_0_useAge_True_t023358/REPETITION_5/TB_fold_3/last.pt"
]

SAVE_PATH = os.path.join(
    os.getcwd(),
    "GradCAM_evaluation",
    datetime.now().strftime("%Y%m%d"),
    MR_MODALITY_TO_PLOT,
    f"Augmentation_{USE_AUGMENTATION}",
)
pathlib.Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

# %% LOOP THROUGH ALL THE MODELS AND GET GRADCAMS

for PATH_TO_MODEL_CKTP in PATH_TO_MODELs_CKTP:
    PATH_TO_MODEL_CKTP = pathlib.Path(PATH_TO_MODEL_CKTP)
    FOLD_NAME = PATH_TO_MODEL_CKTP.parts[-2].replace("TB_", "")
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

    # %% LOAD MODEL
    net = torch.load(PATH_TO_MODEL_CKTP)
    if "vit" in MODEL_VERSION.lower():
        if USE_AGE:
            net = net.to(device)
        else:
            net = net.model.model.to(device)
    else:
        if USE_AGE:
            net = net.to(device)
        else:
            net = net.model.to(device)

    # %% TESTING THINGS
    # import timm
    # from timm.models.layers import PatchEmbed

    # from torchvision.models.feature_extraction import get_graph_node_names
    # from pprint import pprint

    # model = torch.load(PATH_TO_MODEL_CKTP)
    # model = model.model.to(device)
    # nodes, _ = get_graph_node_names(model, tracer_kwargs={"leaf_modules": [PatchEmbed]})
    # pprint(nodes)

    # from torchvision.models.feature_extraction import create_feature_extractor

    # N = 11

    # # This is the "one line of code" that does what you want
    # feature_extractor = create_feature_extractor(
    #     model,
    #     return_nodes=[f"model.encoder.layers.encoder_layer_{N}.self_attention"],
    #     tracer_kwargs={"leaf_modules": [PatchEmbed]},
    # )

    # with torch.no_grad():
    #     out = feature_extractor(model_input[0])

    # print(out[f"model.encoder.layers.encoder_layer_{N}.self_attention"][0].shape)

    # attn_output = out[f"model.encoder.layers.encoder_layer_{N}.self_attention"][0][0]
    # attn_output_weights = out[f"model.encoder.layers.encoder_layer_{N}.self_attention"][
    #     0
    # ][1]

    # print("Attention output", attn_output.shape)
    # print("Attention weights", attn_output_weights.shape)

    # %% LOOP THROUGH ALL THE SETS_TO_PLOT
    # load the datasetsplit .csv file
    importlib.reload(dataset_utilities)
    dataset_split = pd.read_csv(PATH_TO_SPLIT_CSV)
    try:
        dataset_split = dataset_split.drop(columns=["level_0", "index"])
    except:
        pass

    # fix dataset paths if the model has been trained on a different model
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
            transforms.RandomResizedCrop(224, scale=(0.5, 1.5), ratio=(0.7, 1.33)),
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
            transforms.Normalize(np.array(IMG_MEAN_NORM), np.array(IMG_STD_NORM)),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(np.array(IMG_MEAN_NORM), np.array(IMG_STD_NORM)),
        ]
    )

    for set_to_plot in SETS_TO_PLOT:
        print(f"Working on {set_to_plot} set ({MODEL_VERSION}, fold: {FOLD_NAME}).")
        # create save folder in the model repetition folder
        save_path = pathlib.Path(
            os.path.join(
                SAVE_PATH,
                MODEL_NAME,
                set_to_plot,
            )
        )
        save_path.mkdir(parents=True, exist_ok=True)

        # create datagenerator for teh given set
        g = torch.Generator()
        g.manual_seed(training_config.training_settings.random_state)

        samples = dataset_split.loc[
            (dataset_split[FOLD_NAME] == set_to_plot)
            & (dataset_split["target"].isin(CLASSES_TO_USE))
        ]
        print(f"   Nbr. {set_to_plot} files: {len(samples)}")

        # fix the missing age normalizad values for the test set
        if np.isnan(list(samples["age_normalized"])[0]):
            print("Fixing normalized age in the testing set...")
            # retrieve the age in days from the file name
            test_ages = np.array(samples["age_in_days"])
            # get the mean and std deviation from the training cases in this fold
            train_ages = np.array(
                dataset_split.loc[dataset_split[f"fold_{FOLD}"] == "training"][
                    "age_in_days"
                ]
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
                transform=train_transform if USE_AUGMENTATION else validation_transform,
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
            64,
            num_workers=15,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # ## APPLY GRADCAM
        # define target layer
        if "vit" in MODEL_VERSION.lower():
            if USE_AGE:
                target_layers = [net.model.model.encoder.layers[-1].ln_1]
            else:
                target_layers = [net.encoder.layers[-1].ln_1]
            # target_layers = [net.encoder.ln]
            use_reshape_transfor = True
        else:
            if USE_AGE:
                target_layers = [net.model.model.layer4[-1].conv3]
            else:
                target_layers = [net.model.layer4[-1].conv3]
            use_reshape_transfor = False
        print(f"   Using {target_layers} as target layer")

        # get batch from sample_dataloader
        dataiter = iter(sample_dataloader)
        batch = next(dataiter)
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

        accuracy = np.sum(labels == y_) / labels.shape[0]
        from sklearn.metrics import matthews_corrcoef

        mcc = matthews_corrcoef(labels, y_)
        print(
            f"   Perfonace on a batch on {set_to_plot} set: MCC: {mcc:0.3f}, Accuracy: {accuracy:0.3f}"
        )

        print("   Computing GradCAMs")
        cam = EigenCAM(
            model=net,
            target_layers=target_layers,
            use_cuda=False,
            reshape_transform=reshape_transform if use_reshape_transfor else None,
        )

        # compute GradCAM for all the images in the batch
        targets = [ClassifierOutputTarget(c) for c in list(y_)]
        grayscale_cam = cam(
            input_tensor=model_input,
            targets=targets,
            aug_smooth=False,
            eigen_smooth=True,
        )

        # ## PLOT and SAVE gradcams
        # here we save samples_per_image iamges at the time (forst row gray scale image and second row the gradcam)
        samples_per_image = 8
        index_start = range(0, images.shape[0], samples_per_image)
        index_end = range(samples_per_image, images.shape[0] + 1, samples_per_image)

        for idx_img, (idx_s, idx_e) in enumerate(zip(index_start, index_end)):
            print(
                f"   Saving figure {idx_img+1:4d}/{images.shape[0]//samples_per_image}\r",
                end="",
            )
            # take out gray scale images and gradcam images
            g_images = images[idx_s:idx_e, :, :]
            c_iamges = grayscale_cam[idx_s:idx_e, :, :]

            # build image
            c_aus, vis_aus = [], []
            for i in range(g_images.shape[0]):
                g_images_rgb = np.transpose(g_images.numpy()[i], (1, 2, 0))
                # un-normalize (undo T.Normalize)
                g_images_rgb = g_images_rgb * np.array(IMG_MEAN_NORM) + np.array(
                    IMG_STD_NORM
                )
                # bring in [0,1]
                g_images_rgb = (g_images_rgb - g_images_rgb.min()) / (
                    g_images_rgb.max() - g_images_rgb.min()
                )

                # # add label to the image (G=ground truth and P=prediction)
                # img = PIL.Image.fromarray(
                #     np.uint8(cm.gist_earth(g_images_rgb[:, :, 0]) * 255)
                # )
                # text = f"G:{labels[i]:01d}, P:{y_[i]:01d}"
                # draw = ImageDraw.Draw(img)
                # font = ImageFont.truetype(
                #     "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 20
                # )
                # draw.text((0, 0), text, (255, 255, 255), font=font)
                # # convert back
                # g_images_rgb = np.array(img, dtype=np.float32)[:, :, 0] / 255
                # g_images_rgb = np.transpose(
                #     np.stack([g_images_rgb, g_images_rgb, g_images_rgb]), (1, 2, 0)
                # )

                visualization = show_cam_on_image(
                    g_images_rgb, c_iamges[i], use_rgb=True
                )
                # save for making grid
                c_aus.append(torch.tensor(g_images_rgb.transpose(2, 0, 1)))
                vis_aus.append(torch.tensor((visualization / 255).transpose(2, 0, 1)))

            fig = plt.figure(figsize=(30, 30))
            c_aus.extend(vis_aus)
            plt.imshow(
                torchvision.utils.make_grid(
                    c_aus, nrow=samples_per_image, normalize=False
                )
                .numpy()
                .transpose(1, 2, 0)
            )
            plt.axis("off")

            # save
            plt.savefig(
                fname=os.path.join(
                    save_path,
                    f"{FOLD_NAME}_{set_to_plot}_gradCAM_img_{idx_img:03d}.pdf",
                ),
                dpi=100,
                format="pdf",
                bbox_inches="tight",
            )
            plt.savefig(
                fname=os.path.join(
                    save_path,
                    f"{FOLD_NAME}_{set_to_plot}_gradCAM_img_{idx_img:03d}.png",
                ),
                dpi=100,
                format="png",
                bbox_inches="tight",
            )
            plt.close(fig)

# %%
