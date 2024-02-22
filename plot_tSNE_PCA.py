# %%
"""
Script that runs a bunch of classifiers on the features extracted from the trained model (just the encoder part).

Steps:
1 -  define which model to evaluate
2 - load the dataset split .csv file 
3 - for every fold:
    - extract features from the augmented training data using the pre-trained model. 
        If age is used, the image and age features are extracted and PCA is performed on the concatenated vector of image and age features.
    - apply PCA to the extracted features
    - for every classifier:
        - fit classifier to the training data
        - get performance scores on the validation and training sets. Do per-slice and per-subject evaluation (aggregation of the per-slice predictions).
        - (add some plots)
        - save performances in a shared file
4 - aggregate results from each fold

some of the code originates from 
https://scikit-learn.org/0.15/auto_examples/plot_classifier_comparison.html

"""

import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms as T
import numpy as np
import cv2
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import importlib
from scipy.special import softmax

from omegaconf import OmegaConf

import dataset_utilities
import model_bucket_CBTN_v1
from copy import deepcopy
import pathlib

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

import glob


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef

pl.seed_everything(42)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

import evaluation_utilities

# %% LOCAL UTILITIES


@torch.no_grad()
def get_features(
    image_feature_extractor,
    data_loader,
    use_age: bool = False,
    age_feature_extractor=None,
    nbr_images_to_generate: int = None,
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
):
    # # Encode as many images as requested
    if use_age:
        feats, labels, ages = [], [], []
    else:
        feats, labels = [], []
    nbr_images_to_generate = (
        nbr_images_to_generate if nbr_images_to_generate else len(data_loader.dataset)
    )
    counter = 0
    while counter < nbr_images_to_generate:
        for idx, batch in enumerate(data_loader):
            if use_age:
                batch_img_input = batch[0].to(device)
                batch_age_input = batch[2].to(device)
                batch_labels = batch[1]
                ages.extend(batch[2].numpy())
                # encode the age
                batch_age_feats = age_feature_extractor(batch_age_input)
            else:
                batch_img_input = batch[0].to(device)
                batch_labels = batch[1]

            batch_img_feats = image_feature_extractor(batch_img_input)
            if use_age:
                # concatenate image and age features, and save
                batch_img_age_feats = torch.hstack([batch_img_feats, batch_age_feats])
                feats.extend(batch_img_age_feats.detach().cpu().numpy())
            else:
                feats.extend(batch_img_feats.detach().cpu().numpy())
            labels.extend(batch_labels.numpy())

            # update counter
            counter += batch_labels.shape[0]
            print(
                f"(feature extraction) Processing {counter}\{nbr_images_to_generate}\r",
                end="",
            )
            if counter >= nbr_images_to_generate:
                break
    print("\n")

    if use_age:
        return np.stack(feats), np.stack(labels), np.stack(ages)
    else:
        return np.stack(feats), np.stack(labels)


@torch.no_grad()
def get_original_model_prediction(
    model,
    data_loader,
    use_age: bool = False,
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
):
    # # Encode as many images as requested
    if use_age:
        pred, labels, age = [], [], []
    else:
        pred, labels = [], []

    for idx, batch in enumerate(data_loader):
        if use_age:
            model_input = (batch[0].to(device), batch[2].to(device))
            batch_labels = batch[1]
        else:
            model_input = batch[0].to(device)
            batch_labels = batch[1]

        batch_feats = model(model_input)
        pred.extend(batch_feats.detach().cpu().numpy())
        labels.extend(batch_labels.numpy())
        print(
            f"(original DL classification) Processing {idx+1}\{len(data_loader)}\r",
            end="",
        )

    print("\n")
    return np.stack(pred), np.stack(labels)


def plot_embeddings(
    emb,
    hue_labels,
    style_labels=None,
    tool="tsne",
    draw: bool = True,
    save_figure: str = None,
    save_path: str = None,
    prefix: str = "Embeddings_cluster",
):
    if tool == "tsne":
        tl = TSNE(n_components=3, perplexity=int(emb.shape[0] / 6))
    else:
        tl = PCA(n_components=3)
    embedding = tl.fit_transform(emb)

    # define hue order (to keep plots the same)
    hue_order = list(dict.fromkeys(hue_labels))
    print(hue_order)
    # create axis
    fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # populate axis
    for idx, (ax, dim_indexes, view_names) in enumerate(
        zip(
            fig.axes,
            ([0, 1], [0, 2], [1, 2]),
            ("dim_1 - vs - dim_2", "dim_1 - vs - dim_3", "dim_2 - vs - dim_3"),
        )
    ):
        sns.scatterplot(
            x=embedding[:, dim_indexes[0]],
            y=embedding[:, dim_indexes[1]],
            hue=hue_labels,
            hue_order=hue_order,
            style=style_labels,
            legend=False if idx != 2 else True,
            ax=ax,
        )
        # set title
        ax.set_title(f"{tool.upper()} ({view_names})")

        # remove legend for all apart from last plot
        if idx == 2:
            lgnd = ax.legend(
                loc="center left", ncol=3, bbox_to_anchor=(1.1, 0.5), fontsize=5
            )
            # plt.setp(ax.get_legend().get_texts(), fontsize="5")
            for markers in lgnd.legendHandles:
                markers._sizes = [10]

    # hide last axis
    axis[1, 1].axis("off")

    if save_figure:
        fig.savefig(
            os.path.join(save_path, f"{prefix}_{tool.upper()}.pdf"),
            dpi=100,
            bbox_inches="tight",
        )
        fig.savefig(
            os.path.join(save_path, f"{prefix}_{tool.upper()}.png"),
            dpi=100,
            bbox_inches="tight",
        )
    if draw:
        plt.show()
    else:
        plt.close(fig)


def plot_PCA_embeddings(
    embedding,
    hue_labels,
    style_labels=None,
    draw: bool = True,
    save_figure: str = None,
    save_path: str = None,
    prefix: str = "Embeddings_cluster",
    nbr_legend_columns: int = 3,
    value_ranges=None,
):
    # define hue order
    hue_order = list(dict.fromkeys(hue_labels))
    hue_order.sort()
    # create axis
    fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # populate axis
    for idx, (ax, dim_indexes, view_names) in enumerate(
        zip(
            fig.axes,
            ([0, 1], [0, 2], [1, 2]),
            ("dim_1 - vs - dim_2", "dim_1 - vs - dim_3", "dim_2 - vs - dim_3"),
        )
    ):
        sns.scatterplot(
            x=embedding[:, dim_indexes[0]],
            y=embedding[:, dim_indexes[1]],
            hue=hue_labels,
            hue_order=hue_order,
            style=style_labels,
            legend=False if idx != 2 else True,
            ax=ax,
        )
        # set axis limits
        if value_ranges:
            ax.set_xlim(
                (value_ranges[dim_indexes[0]][0], value_ranges[dim_indexes[0]][1])
            )
            ax.set_ylim(
                (value_ranges[dim_indexes[1]][0], value_ranges[dim_indexes[1]][1])
            )

        # set title
        ax.set_title(f"PCA ({view_names})")

        # remove legend for all apart from last plot
        if idx == 2:
            # ax.legend(
            #     loc="center left", ncol=nbr_legend_columns, bbox_to_anchor=(1.1, 0.5)
            # )
            # plt.setp(ax.get_legend().get_texts(), fontsize="5")

            # remove legend for all apart from last plot
            lgnd = ax.legend(
                loc="center left", ncol=3, bbox_to_anchor=(1.1, 0.5), fontsize=5
            )
            # plt.setp(ax.get_legend().get_texts(), fontsize="5")
            for markers in lgnd.legendHandles:
                markers._sizes = [10]

    # hide last axis
    axis[1, 1].axis("off")

    if save_figure:
        fig.savefig(
            os.path.join(save_path, f"{prefix}.pdf"),
            dpi=100,
            bbox_inches="tight",
        )
        fig.savefig(
            os.path.join(save_path, f"{prefix}.png"),
            dpi=100,
            bbox_inches="tight",
        )
    if draw:
        plt.show()
    else:
        plt.close(fig)


def one_hot_to_class_string(one_hot_encodigs, target_class_to_one_hot_mapping):
    str_labels = []
    for l in one_hot_encodigs:
        for k, v in target_class_to_one_hot_mapping.items():
            if all(v == torch.tensor(l)):
                str_labels.append(k)
    return str_labels


def make_meshgrid(x, y, h=0.5):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def get_age_from_file_name(file_name):
    return int(
        os.path.basename(file_name).split("___")[3].split("_")[0].replace("d", "")
    )


def fix_not_normalized_age(datasplit_df, samples_df, fold_nbr):
    # fix the missing age normalizad values for the test set
    if np.isnan(list(samples_df["age_normalized"])[0]):
        print("Fixing normalized age in the testing set...")
        # retrieve the age in days from the file name
        test_ages = np.array(samples_df["age_in_days"])
        # get the mean and std deviation from the training cases in this fold
        train_ages = np.array(
            datasplit_df.loc[datasplit_df[f"fold_{fold_nbr}"] == "training"][
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
        samples_df.loc[:, "age_normalized"] = test_ages_normalized.tolist()
    return samples_df


# %% PATHS AND SETTINGS

PATH_TO_LOCAL_DATASET_CONFIGS = (
    "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/conf/dataset"
)

# collect paths to models from the given folder
PATH_TO_TRAINED_MODELS = "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/train_model_archive_POST_20231208"


# settings
# SETS_TO_PLOT = ["training", "validation", "test"]
USE_AUGMENTATION = False
MR_MODALITY_TO_PLOT = ["ADC", "T1", "T2"]
REPETITION = 5
FOLD = 3
PATH_TO_MODELs_CKTP = []

for mr_sequence in MR_MODALITY_TO_PLOT:
    for mr_sequence_folder in glob.glob(os.path.join(PATH_TO_TRAINED_MODELS, "*", "")):
        if mr_sequence in pathlib.Path(mr_sequence_folder).parts[-1]:
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
                    PATH_TO_MODELs_CKTP.append(ckpt_path)

    PATH_TO_MODELs_CKTP = [
        "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/train_model_archive_POST_20231208/ADC_TESTs_20240128/ViT_b_16_pretrained_True_SimCLR_dataset_TCGA_frozen_True_0.5_LR_1e-05_BATCH_128_AUGMENTATION_True_OPTIM_adam_SCHEDULER_exponential_MLPNODES_0_useAge_True_t181432/REPETITION_5/TB_fold_3/last.pt"
    ]

    # where to save the evaluation
    SAVE_PATH = os.path.join(
        os.getcwd(),
        "PCA_evaluation",
        datetime.now().strftime("%Y%m%d"),
        mr_sequence,
        f"Augmentation_{USE_AUGMENTATION}",
    )
    pathlib.Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

    # PCA settings
    PCA_NBR_COMPONENTS = 2

    # %% LOOP THROUGH ALL THE MODELS
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

        # change the path to the files in the data_split.csv file
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

        # build mapping between class and one hot encoding
        # get mapping of the classes to one hot encoding
        unique_targe_classes = list(pd.unique(dataset_split["target"]))
        unique_targe_classes.sort()
        unique_targe_classes = dict.fromkeys(unique_targe_classes)

        one_hot_encodings = torch.nn.functional.one_hot(
            torch.tensor(list(range(len(unique_targe_classes))))
        )

        target_class_to_one_hot_mapping = dict(
            zip(unique_targe_classes, one_hot_encodings)
        )

        # define data loader transforms
        train_transform = T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.5, 1.5), ratio=(0.7, 1.33)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(45),
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                        )
                    ],
                    p=0.5,
                ),
                T.ToTensor(),
                T.Normalize(np.array(IMG_MEAN_NORM), np.array(IMG_STD_NORM)),
            ]
        )

        validation_transform = T.Compose(
            [
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(np.array(IMG_MEAN_NORM), np.array(IMG_STD_NORM)),
            ]
        )

        # %% LOAD MODEL
        full_model = torch.load(PATH_TO_MODEL_CKTP)
        image_feature_extractor = deepcopy(full_model.model.model)
        age_feature_extractor = deepcopy(full_model.age_encoder) if USE_AGE else None

        # remove classification head to only get the features
        if "resnet" in MODEL_VERSION.lower():
            image_feature_extractor.fc = torch.nn.Identity()
        elif "vit" in MODEL_VERSION.lower():
            image_feature_extractor.heads = torch.nn.Identity()

        full_model.to(device)
        full_model.eval()
        image_feature_extractor.to(device)
        image_feature_extractor.eval()
        if age_feature_extractor:
            age_feature_extractor.to(device)
            age_feature_extractor.eval()

        # %% OBTAIN TRAINING AND VALIDATION FEATURES
        set_features = {"training": [], "validation": [], "test": []}

        for set_name in set_features.keys():
            print(f"Working on {set_name} set...")

            # create dataset for this set
            samples = dataset_split.loc[dataset_split[FOLD_NAME] == set_name]
            # fix normalized ages if needed
            samples = fix_not_normalized_age(dataset_split, samples, FOLD)

            print(f"{set_name.title()}: {len(samples)} nbr. files")

            # build the data loader
            sample_dataloader = DataLoader(
                dataset_utilities.PNGDatasetFromFolder(
                    list(samples["file_path"]),
                    transform=(
                        train_transform if USE_AUGMENTATION else validation_transform
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
                64,
                num_workers=15,
                shuffle=True,
                worker_init_fn=None,
                generator=None,
            )

            set_feats = get_features(
                image_feature_extractor,
                sample_dataloader,
                nbr_images_to_generate=None,
                use_age=USE_AGE,
                age_feature_extractor=age_feature_extractor,
            )

            # save
            set_features[set_name] = set_feats

        # %% APPLY PCA ON THE TRAINING SET AND, USING THE SAME TRANSFORMATION, ON THE VALIDATION SET
        # work on plotting first (always PCA components = 3)
        pca = PCA(n_components=3)
        pca_training = pca.fit_transform(set_features["training"][0])
        # get the ranges of the different PCA components to have the plots scaled the same way
        PCA_plot_ranges = [
            (np.min(pca_training[:, i]), np.max(pca_training[:, i])) for i in range(3)
        ]

        # create save path for this model and save plots
        save_path = os.path.join(SAVE_PATH, MODEL_NAME)
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

        for set_name in set_features.keys():
            # project using training
            pca_proj = pca.transform(set_features[set_name][0])

            # plot first 3 dimensions
            plot_PCA_embeddings(
                pca_proj,
                hue_labels=one_hot_to_class_string(
                    set_features[set_name][1], target_class_to_one_hot_mapping
                ),
                style_labels=list(
                    dataset_split.loc[dataset_split[FOLD_NAME] == set_name][
                        "subject_IDs"
                    ]
                ),
                draw=False,
                save_figure=True,
                save_path=save_path,
                prefix=f"PCA_{set_name}_rep_{REPETITION}_fold_{FOLD}",
                nbr_legend_columns=4,
                value_ranges=PCA_plot_ranges,
            )

# %%
