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
from datetime import datetime
import sys
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
from itertools import cycle

from omegaconf import OmegaConf

import dataset_utilities
import model_bucket_CBTN_v1
from copy import deepcopy
import pathlib

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

import glob

import torchvision


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
def get_per_subject_features_MIL_organized(
    df_samples,
    image_feature_extractor,
    transform=None,
    use_age: bool = False,
    use_normalized_age: bool = False,
    age_feature_extractor=None,
    nbr_of_subjects_to_generate: int = None,
    nbr_images_to_generate_per_subject: int = None,
    labels_as_one_hot: bool = True,
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
):
    """
    Different from the get_feature function, here we get as many images per patient as requested using data augmentation.
    One can generate as many augmented subjects as needed. This function also organizes the features in bags as needed for the CLAM MIL training that
    is a list of tuples where the first element of each tuple is a np array of the features (N x K where N is the number of instances and K is the feature embedding space)
    and the second element is the label.
    """

    # get the number of subjects to generate and the number of instances for each subject
    if nbr_of_subjects_to_generate is None:
        nbr_of_subjects_to_generate = len(df_samples.subject_IDs.unique())

    if nbr_images_to_generate_per_subject is None:
        nbr_images_to_generate_per_subject = int(
            np.mean(df_samples.subject_IDs.value_counts())
        )

    # create a cycle of all the subjects to allow for multiple iterations if needed
    unique_subjects = cycle(list(df_samples.subject_IDs.unique()))

    # this loops thorugh all the bags
    bags_of_instances = []

    for s in range(nbr_of_subjects_to_generate):
        # the subject ID from the unique IDs
        subject_ID = next(unique_subjects)
        samples = df_samples.loc[df_samples.subject_IDs == subject_ID]
        sample_label = list(samples["one_hot_encodig"])[0]

        # build a generator only on this subject
        datagen = DataLoader(
            dataset_utilities.PNGDatasetFromFolder(
                list(samples["file_path"]),
                transform=transform,
                labels=[torch.tensor(v) for v in list(samples["one_hot_encodig"])],
                return_file_path=False,
                return_age=use_age,
                ages=list(samples["age_normalized"])
                if use_normalized_age
                else list(samples["age_in_days"]),
            ),
            num_workers=15,
            batch_size=5,
        )

        # get as many images as requested. if not take as many as the mean number of images over the subjects
        instance_counter = 0

        feats = []

        # this loops through all the instances
        while instance_counter < nbr_images_to_generate_per_subject:
            for idx, batch in enumerate(datagen):
                if use_age:
                    batch_img_input = batch[0].to(device)
                    batch_age_input = batch[2].to(device)
                    # encode the age
                    batch_age_feats = age_feature_extractor(batch_age_input)
                else:
                    batch_img_input = batch[0].to(device)

                batch_img_feats = image_feature_extractor(batch_img_input)
                if use_age:
                    # concatenate image and age features, and save
                    batch_img_age_feats = torch.hstack(
                        [batch_img_feats, batch_age_feats]
                    )
                    feats.extend(batch_img_age_feats.detach().cpu().numpy())
                else:
                    feats.extend(batch_img_feats.detach().cpu().numpy())

                # update counter
                instance_counter += batch_img_input.shape[0]
                print(
                    f"(feature extraction) Processing bag {s+1}\{nbr_of_subjects_to_generate}, instance {instance_counter}\{nbr_images_to_generate_per_subject}\r",
                    end="",
                )
                if instance_counter >= nbr_images_to_generate_per_subject:
                    break

        # add this features to the bags_of_instances
        bags_of_instances.append(
            (
                np.stack(feats),
                np.array(sample_label)
                if labels_as_one_hot
                else np.expand_dims(np.argmax(sample_label), axis=0),
            )
        )
    print("\n")

    return bags_of_instances


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


# %% LOOP ON MANY MODELS and FOLDS
# model_save_path = "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/trained_model_archive/TESTs_20231204"
# models_to_evaluate = []
# for model_run in glob.glob(os.path.join(model_save_path, "*", "")):
#     # for all the repetiitons
#     for repetition in glob.glob(os.path.join(model_run, "REPETITION*", "")):
#         models_to_evaluate.append(repetition)

# print(f"Found {len(models_to_evaluate)} to work on.")

models_to_evaluate = [
    "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/trained_model_archive/TESTs_20231206/ResNet50_pretrained_True_ImageNet_dataset_ImageNet_frozen_True_0.5_LR_1e-05_BATCH_128_AUGMENTATION_True_OPTIM_adam_SCHEDULER_exponential_MLPNODES_0_useAge_True_t094815/REPETITION_1"
]

for idy, model_path in enumerate(models_to_evaluate):
    MODEL_INFO = {
        "pretraining_type": "classification",  # or SimCLR
        "path": model_path,
        "hydra_config_path": os.path.join(model_path, "hydra_config.yaml"),
        "CLAM_writer_dir": [],
    }
    DATASET_INFO = {
        "dataset_split_path": os.path.join(
            MODEL_INFO["path"], "data_split_information.csv"
        ),
        "nbr_training_samples_to_embed": None,  # this is an augmented version fo the training set.
    }

    # load training configuration and get information about the model
    training_config = OmegaConf.load(
        MODEL_INFO["hydra_config_path"],
    )

    model_version = training_config.model_settings.model_version
    use_age = training_config.dataloader_settings.use_age
    use_normalized_age = training_config.dataloader_settings.normalize_age
    session_time = training_config.logging_settings.start_time
    repetition = str(os.path.basename(pathlib.Path(MODEL_INFO["path"])))

    SAVE_PATH = pathlib.Path(
        os.path.join(MODEL_INFO["path"], "Feature_classification_experiments")
    )
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_INFO.update(
        {
            "CLAM_writer_dir": os.path.join(
                os.getcwd(),
                "trained_model_archive",
                f'CLAM_TESTs_{datetime.now().strftime("%Y%m%d")}',
                "_".join([model_version, "AGE", str(use_age)]),
                f"REPETITION_{repetition}",
                f'{datetime.now().strftime("t%H%M%S")}',
            )
        }
    )
    pathlib.Path(MODEL_INFO["CLAM_writer_dir"]).mkdir(parents=True, exist_ok=True)

    # %% INITIATE SUMMARY PERFORMANCE
    """
    Initialize where classification summary are saved
    - model_version
    - repetition
    - nbr_classes
    - classes
    - dataset_version
    - fold_nbr
    - set
    - nbr_PCA_components
    """
    summary_performance = []

    # %% BUILD DATASETS
    # load dataset_split_infromation
    print(DATASET_INFO["dataset_split_path"])
    dataset_split = pd.read_csv(DATASET_INFO["dataset_split_path"])
    try:
        dataset_split = dataset_split.drop(columns=["level_0", "index"])
    except:
        print()

    # get mapping of the classes to one hot encoding
    unique_targe_classes = list(pd.unique(dataset_split["target"]))
    unique_targe_classes.sort()
    unique_targe_classes = dict.fromkeys(unique_targe_classes)

    one_hot_encodings = torch.nn.functional.one_hot(
        torch.tensor(list(range(len(unique_targe_classes))))
    )
    # build mapping between class and one hot encoding
    target_class_to_one_hot_mapping = dict(zip(unique_targe_classes, one_hot_encodings))

    print(target_class_to_one_hot_mapping)

    # add age infromation for all the files in the dataset
    dataset_split["age"] = [
        get_age_from_file_name(f) for f in list(dataset_split["file_path"])
    ]

    # loop through all the available folds
    for fold_idx, fold in enumerate(
        glob.glob(os.path.join(MODEL_INFO["path"], "TB_fold*", ""))
    ):
        fold_idx = int(os.path.basename(pathlib.Path(fold)).split("_")[-1])
        print(
            f"\nModel {idy+1} of {len(models_to_evaluate)} ({model_version} {repetition} {fold_idx} {session_time})"
        )
        # % BUILD TRAINING AND VALIDATION DATASETS
        # ######### training set
        training_files_for_inference = dataset_split.loc[
            dataset_split[f"fold_{fold_idx}"] == "training"
        ].reset_index()
        # make torch tensor labels
        training_labels = [
            target_class_to_one_hot_mapping[c]
            for c in list(training_files_for_inference["target"])
        ]
        training_labels_for_df = [list(l.numpy()) for l in training_labels]
        # add label to the dataframe
        training_files_for_inference.insert(
            training_files_for_inference.shape[1],
            "one_hot_encodig",
            training_labels_for_df,
        )

        # print summary of files
        [
            print(
                f"{c}: {len(training_files_for_inference.loc[training_files_for_inference['target']==c])}"
            )
            for c in list(pd.unique(training_files_for_inference["target"]))
        ]

        # ######### validation set
        validation_files_for_inference = dataset_split.loc[
            dataset_split[f"fold_{fold_idx}"] == "validation"
        ].reset_index()
        # make torch tensor labels
        validation_labels = [
            target_class_to_one_hot_mapping[c]
            for c in list(validation_files_for_inference["target"])
        ]
        validation_labels_for_df = [list(l.numpy()) for l in validation_labels]
        # add label to the dataframe
        validation_files_for_inference.insert(
            validation_files_for_inference.shape[1],
            "one_hot_encodig",
            validation_labels_for_df,
        )

        [
            print(
                f"{len(validation_files_for_inference.loc[validation_files_for_inference['target']==c]) for c in list(pd.unique(validation_files_for_inference['target']))}"
            )
        ]

        preprocess = T.Compose(
            [
                T.Resize(
                    size=list(training_config.dataloader_settings.input_size),
                    antialias=True,
                ),
                T.ToTensor(),
                T.Normalize(
                    mean=list(training_config.dataloader_settings.img_mean),
                    std=list(training_config.dataloader_settings.img_std),
                ),
            ],
        )
        transforms = T.Compose(
            [
                T.RandomResizedCrop(
                    size=training_config["dataloader_settings"]["input_size"],
                    scale=(0.6, 1.5),
                    ratio=(0.75, 1.33),
                    antialias=True,
                ),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(45),
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                        ),
                        T.GaussianBlur(kernel_size=9),
                    ],
                    p=0.8,
                ),
                T.RandomApply(
                    [T.TrivialAugmentWide(num_magnitude_bins=15, fill=-1)], p=1
                ),
                T.ToTensor(),
                T.Normalize(
                    mean=list(training_config.dataloader_settings.img_mean),
                    std=list(training_config.dataloader_settings.img_std),
                ),
            ],
        )

        # # build adataloaders
        # training_dataset = DataLoader(
        #     dataset_utilities.PNGDatasetFromFolder(
        #         list(training_files_for_inference["file_path"]),
        #         transform=preprocess,
        #         labels=training_labels,
        #         return_file_path=False,
        #         return_age=use_age,
        #         ages=list(training_files_for_inference["age_normalized"])
        #         if use_normalized_age
        #         else list(training_files_for_inference["age_in_days"]),
        #     ),
        #     num_workers=15,
        #     batch_size=32,
        # )

        # validation_dataset = DataLoader(
        #     dataset_utilities.PNGDatasetFromFolder(
        #         list(validation_files_for_inference["file_path"]),
        #         transform=preprocess,
        #         labels=validation_labels,
        #         return_file_path=False,
        #         return_age=use_age,
        #         ages=list(validation_files_for_inference["age_normalized"])
        #         if use_normalized_age
        #         else list(validation_files_for_inference["age_in_days"]),
        #     ),
        #     num_workers=15,
        #     batch_size=32,
        # )

        # %% LOAD THIS FOLD"S MODEL
        class ResNetFeatureExtractor(torch.nn.Module):
            def __init__(self, model):
                super(ResNetFeatureExtractor, self).__init__()
                self.part = torch.nn.Sequential(
                    model.conv1,
                    model.bn1,
                    model.relu,
                    model.maxpool,
                    model.layer1,
                    model.layer2,
                    model.layer3,
                    torch.nn.AdaptiveAvgPool2d(1),
                )

            def forward(self, x):
                x = self.part(x)
                x = x.view(x.size(0), -1)
                return x

        # load ImageNet ResNet50
        from torchvision import models

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        image_feature_extractor = ResNetFeatureExtractor(resnet)
        # model_path = os.path.join(fold, "last.pt")
        # if MODEL_INFO["pretraining_type"] == "classification":
        #     original_model = torch.load(model_path)
        #     # image_feature_extractor = deepcopy(original_model.model.model)
        #     image_feature_extractor = ResNetFeatureExtractor(original_model.model.model)
        #     age_feature_extractor = (
        #         deepcopy(original_model.age_encoder) if use_age else None
        #     )
        # elif MODEL_INFO["pretraining_type"] == "SimCLR":
        #     original_model = (
        #         model_bucket_CBTN_v1.SimCLRModelWrapper.load_from_checkpoint(
        #             model_path,
        #         )
        #     )
        #     original_model = deepcopy(original_model.convnet)
        #     image_feature_extractor = deepcopy(original_model)

        # # remove classification head to only get the features
        # if "resnet" in model_version.lower():
        #     if model_version.lower() == "resnet9":
        #         image_feature_extractor.classifier[2] = torch.nn.Identity()
        #     else:
        #         image_feature_extractor.fc = torch.nn.Identity()
        # elif "vit" in model_version.lower():
        #     image_feature_extractor.heads = torch.nn.Identity()

        # original_model.to(device)
        # original_model.eval()
        image_feature_extractor.to(device)
        image_feature_extractor.eval()
        # if age_feature_extractor:
        #     age_feature_extractor.to(device)
        #     age_feature_extractor.eval()

        # # %% DEBUG BUILD TRAINING DATALOADER
        # importlib.reload(dataset_utilities)

        # training_dataloader = dataset_utilities.CustomDataset(
        #     train_sample_paths=list(training_files_for_inference["file_path"]),
        #     validation_sample_paths=list(validation_files_for_inference["file_path"]),
        #     test_sample_paths=None,
        #     training_targets=list(training_files_for_inference["target"]),
        #     validation_targets=list(validation_files_for_inference["target"]),
        #     test_targets=None,
        #     batch_size=32,
        #     num_workers=15,
        #     preprocess=preprocess,
        #     transforms=preprocess,
        #     training_batch_sampler=None,
        # )
        # training_dataloader.setup()
        # training_dataset = training_dataloader.train_dataloader()
        # validation_dataset = training_dataloader.val_dataloader()

        # %% OBTAIN ORIGINAL DL CLASSIFIER PREDICTIONS
        # original_DL_classifier_performance = {"training": {}, "validation": {}}

        # for data, data_name in zip(
        #     (training_dataset, validation_dataset), ("training", "validation")
        # ):
        #     pred, labels = get_original_model_prediction(
        #         original_model, data, use_age=use_age, device=device
        #     )
        #     original_DL_classifier_performance[data_name][
        #         f"original_DL_accuracy"
        #     ] = accuracy_score(
        #         np.argmax(labels, axis=1), np.argmax(softmax(pred, axis=1), axis=1)
        #     )
        #     original_DL_classifier_performance[data_name][
        #         f"original_DL_mcc"
        #     ] = matthews_corrcoef(
        #         np.argmax(labels, axis=1), np.argmax(softmax(pred, axis=1), axis=1)
        #     )

        #     # save confusion matrix
        #     evaluation_utilities.plotConfusionMatrix(
        #         GT=labels,
        #         PRED=softmax(pred, axis=1),
        #         classes=list(unique_targe_classes.keys()),
        #         savePath=SAVE_PATH,
        #         saveName=f"Original_model_{data_name}_performance_fold_{fold_idx}",
        #         draw=False,
        #     )

        #     # ## USE TORCH
        #     import torchmetrics

        #     confusion_matrix = torchmetrics.ConfusionMatrix(
        #         task="multiclass",
        #         num_classes=len(list(unique_targe_classes.keys())),
        #         threshold=0.05,
        #     )

        #     confusion_matrix(
        #         torch.argmax(torch.tensor(softmax(pred, axis=1)), dim=1),
        #         torch.argmax(torch.tensor(labels), dim=1),
        #     )
        #     confusion_matrix_computed = (
        #         confusion_matrix.compute().detach().cpu().numpy().astype(int)
        #     )

        #     string_labels = np.unique(list(unique_targe_classes.keys()))
        #     df_cm = pd.DataFrame(
        #         confusion_matrix_computed, index=string_labels, columns=string_labels
        #     )
        #     fig_, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
        #     sns.heatmap(
        #         df_cm, annot=True, cmap="Blues", annot_kws={"size": 20}, ax=ax, fmt="g"
        #     )
        #     plt.xlabel("Predicted")
        #     plt.ylabel("Actual")
        #     fig_.savefig(
        #         os.path.join(
        #             SAVE_PATH,
        #             f"Original_model_{data_name}_performance_fold_{fold_idx}_TORCH.png",
        #         ),
        #         bbox_inches="tight",
        #         dpi=100,
        #     )
        #     plt.close(fig_)

        # if data_name == "training":
        #     [
        #         print(os.path.basename(f), t, l)
        #         for f, t, l in zip(
        #             list(training_files_for_inference["file_path"]),
        #             list(training_files_for_inference["target"]),
        #             labels,
        #         )
        #     ]
        # else:
        #     [
        #         print(os.path.basename(f), t, l)
        #         for f, t, l in zip(
        #             list(validation_files_for_inference["file_path"]),
        #             list(validation_files_for_inference["target"]),
        #             labels,
        #         )
        #     ]

        # %% OBTAIN TRAINING AND VALIDATION FEATURES
        # training_feats = get_features(
        #     image_feature_extractor,
        #     training_dataset,
        #     nbr_images_to_generate=DATASET_INFO["nbr_training_samples_to_embed"],
        #     use_age=use_age,
        #     age_feature_extractor=age_feature_extractor,
        # )

        # validation_feats = get_features(
        #     image_feature_extractor,
        #     validation_dataset,
        #     nbr_images_to_generate=None,
        #     use_age=use_age,
        #     age_feature_extractor=age_feature_extractor,
        # )

        # if use_age:
        #     training_features, training_labels, training_ages = training_feats
        #     validation_features, validation_labels, validation_ages = validation_feats
        # else:
        #     training_features, training_labels = training_feats
        #     validation_features, validation_labels = validation_feats

        # # %% GROUP FEATURES BASED ON THE SUBJECT
        # """
        # The samples for the CLAM model are composed of:
        # - features: torch tensor  NxK where N is the numbner of instances for this bag and K is the size of each instance in the bag.
        # - label: the bag label
        # """

        # def organize_features_in_subjects_bags(
        #     samples_df, features, labels=None, one_hot_labels: bool = True
        # ):
        #     """
        #     Utility that given the damples dataframe and the corresponding extracted features,
        #     returns a list of bags (subjects) where each bag is composed by a list of np arrays of features and the corresponding bag label.
        #     """

        #     list_of_bags = []

        #     for subject in samples_df.subject_IDs.unique():
        #         # get the index in the sample_df for this subjects
        #         subject_indexes = samples_df.index[
        #             samples_df.subject_IDs == subject
        #         ].tolist()
        #         # get all the features from the features array
        #         subject_features = features[subject_indexes, :]
        #         # get label
        #         if labels is None:
        #             # get label from the samples_df
        #             subject_label = samples_df.loc[
        #                 samples_df.subject_IDs == subject
        #             ].target.unique()
        #             if not one_hot_labels:
        #                 subject_label = np.argmax(subject_label, axis=-1)
        #         else:
        #             # get the labels at the first index
        #             subject_label = labels[subject_indexes[0]]
        #             if not one_hot_labels:
        #                 subject_label = np.expand_dims(
        #                     np.argmax(subject_label, axis=-1), axis=-1
        #                 )
        #         # add this subject bag to the list
        #         list_of_bags.append((subject_features, subject_label))

        #     return list_of_bags

        # per_subject_training_samples = organize_features_in_subjects_bags(
        #     training_files_for_inference,
        #     training_features,
        #     training_labels,
        #     one_hot_labels=False,
        # )
        # per_subject_validation_samples = organize_features_in_subjects_bags(
        #     validation_files_for_inference,
        #     validation_features,
        #     validation_labels,
        #     one_hot_labels=False,
        # )
        nbr_instances_per_bag = 25
        per_subject_training_samples = get_per_subject_features_MIL_organized(
            df_samples=training_files_for_inference,
            image_feature_extractor=image_feature_extractor,
            transform=transforms,
            use_age=False,
            nbr_of_subjects_to_generate=1000,
            age_feature_extractor=None,
            nbr_images_to_generate_per_subject=nbr_instances_per_bag,
            labels_as_one_hot=False,
        )

        per_subject_validation_samples = get_per_subject_features_MIL_organized(
            df_samples=validation_files_for_inference,
            image_feature_extractor=image_feature_extractor,
            transform=preprocess,
            use_age=False,
            age_feature_extractor=None,
            nbr_of_subjects_to_generate=None,
            nbr_images_to_generate_per_subject=nbr_instances_per_bag,
            labels_as_one_hot=False,
        )

        # %% DEFINE DATALOADER THAT DIGESTS BATCHES OF BAGS
        import random

        class DatasetFromListOfBags(Dataset):
            def __init__(
                self,
                list_of_bags,
                nbr_instances_per_bag: int = 10,
                shuffle: bool = False,
            ):
                self.list_of_bags = list_of_bags
                self.shuffle = shuffle
                self.nbr_instances_per_bag = nbr_instances_per_bag

                if self.shuffle:
                    random.shuffle(list_of_bags)

            def __getitem__(self, sample_index):
                # select as many samples as specified by nbr_samples_per_bag
                replace = (
                    self.list_of_bags[sample_index][0].shape[0]
                    < self.nbr_instances_per_bag
                )
                index_intances = np.random.choice(
                    a=self.list_of_bags[sample_index][0].shape[0],
                    size=self.nbr_instances_per_bag,
                    replace=replace,
                )
                features = self.list_of_bags[sample_index][0][index_intances, :]
                features = torch.from_numpy(features)
                try:
                    label = torch.from_numpy(self.list_of_bags[sample_index][1])
                except:
                    label = torch.tensor(self.list_of_bags[sample_index][1])
                return features, label

            def __len__(
                self,
            ):
                return len(self.list_of_bags)

        batch_size = 1
        # CLAM_train_dataloader = DataLoader(
        #     DatasetFromListOfBags(
        #         per_subject_training_samples,
        #         nbr_instances_per_bag=nbr_instances_per_bag,
        #         shuffle=True,
        #     ),
        #     batch_size=batch_size,
        #     num_workers=15,
        #     shuffle=True,
        # )
        # CLAM_validation_dataloader = DataLoader(
        #     DatasetFromListOfBags(
        #         per_subject_validation_samples,
        #         nbr_instances_per_bag=nbr_instances_per_bag,
        #         shuffle=False,
        #     ),
        #     batch_size=batch_size,
        #     num_workers=15,
        #     shuffle=False,
        # )

        CLAM_train_dataloader = DatasetFromListOfBags(
            per_subject_training_samples,
            nbr_instances_per_bag=nbr_instances_per_bag,
            shuffle=True,
        )
        CLAM_validation_dataloader = DatasetFromListOfBags(
            per_subject_validation_samples,
            nbr_instances_per_bag=nbr_instances_per_bag,
            shuffle=False,
        )

        x, y = next(iter(CLAM_train_dataloader))
        print(x.shape)
        print(y.shape)

        # %% DEFINE CLAM MODEL
        import model_CLAM

        importlib.reload(model_CLAM)

        clam_classifier = model_CLAM.CLAM_SB(
            instance_feature_size=per_subject_training_samples[0][0].shape[1],
            gate=True,
            size_arg="small",
            dropout=False,
            k_sample=2,
            n_classes=len(training_files_for_inference.target.unique()),
            subtyping=True,
        ).to(device)

        # %% TRAIN
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(MODEL_INFO["CLAM_writer_dir"], flush_secs=15)

        importlib.reload(model_CLAM)

        max_epochs = 100
        learning_rate = 1e-7

        loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([0.6, 1.2, 0.4]).to(device)
        )
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, clam_classifier.parameters()),
            lr=learning_rate,
            weight_decay=1e-5,
        )

        for i in range(max_epochs):
            model_CLAM.train_loop_clam(
                i,
                clam_classifier,
                CLAM_train_dataloader,
                optimizer,
                n_classes=len(training_files_for_inference.target.unique()),
                bag_weight=0.7,
                loss_fn=loss_fn,
                writer=writer,
            )
            model_CLAM.validate_clam(
                1,
                i,
                clam_classifier,
                CLAM_validation_dataloader,
                n_classes=len(training_files_for_inference.target.unique()),
                loss_fn=loss_fn,
                writer=writer,
            )

# %%
