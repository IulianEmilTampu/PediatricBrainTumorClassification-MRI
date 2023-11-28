# %%
"""
Script that runs a bunch of classifiers on the features extracted from the trained model (just the encoder part).

Steps:
1 -  define which model to evaluate
2 - load the dataset split .csv file 
3 - for every fold:
    - extract features from the augmented training data using the pre-trained model
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
    model,
    data_loader,
    nbr_images_to_generate: int = None,
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
):
    # # Encode as many images as requested
    feats, labels = [], []
    nbr_images_to_generate = (
        nbr_images_to_generate if nbr_images_to_generate else len(data_loader.dataset)
    )
    counter = 0
    while counter < nbr_images_to_generate:
        for idx, (batch_imgs, batch_labels) in enumerate(data_loader):
            batch_imgs = batch_imgs.to(device)
            batch_feats = model(batch_imgs)
            feats.extend(batch_feats.detach().cpu().numpy())
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

    return np.stack(feats), np.stack(labels)


@torch.no_grad()
def get_original_model_prediction(
    model,
    data_loader,
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
):
    # # Encode as many images as requested
    pred, labels = [], []

    for idx, (batch_imgs, batch_labels) in enumerate(data_loader):
        batch_imgs = batch_imgs.to(device)
        batch_feats = model(batch_imgs)
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
            ax.legend(loc="center left", ncol=3, bbox_to_anchor=(1.1, 0.5))
            plt.setp(ax.get_legend().get_texts(), fontsize="6")

    # hide last axis
    axis[1, 1].axis("off")

    if save_figure:
        fig.savefig(
            os.path.join(save_path, f"{prefix}_{tool.upper()}.pdf"),
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
        # set title
        ax.set_title(f"PCA ({view_names})")

        # remove legend for all apart from last plot
        if idx == 2:
            ax.legend(
                loc="center left", ncol=nbr_legend_columns, bbox_to_anchor=(1.1, 0.5)
            )
            plt.setp(ax.get_legend().get_texts(), fontsize="5")

    # hide last axis
    axis[1, 1].axis("off")

    if save_figure:
        fig.savefig(
            os.path.join(save_path, f"{prefix}.pdf"),
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


# %% SOME SETTINGS
PCA_NBR_COMPONENTS = 2
names = [
    "Nearest_Neighbors",
    "Linear_SVM",
    "RBF_SVM",
    "Decision_Tree",
    "Random_Forest",
    "AdaBoost",
    "Naive_Bayes",
    "LDA",
    "QDA",
]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA(),
]


# %% LOOP ON MANY MODELS and FOLDS
model_save_path = "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/trained_model_archive/TESTs_20231127"
models_to_evaluate = []
for model_run in glob.glob(os.path.join(model_save_path, "*", "")):
    # for all the repetiitons
    for repetition in glob.glob(os.path.join(model_run, "REPETITION*", "")):
        models_to_evaluate.append(repetition)

print(f"Found {len(models_to_evaluate)} to work on.")

# models_to_evaluate = [
#     "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/trained_model_archive/TESTs_20231127/ResNet50_pretrained_True_frozen_True_0.5_LR_1e-05_BATCH_32_AUGMENTATION_True_OPTIM_adam_SCHEDULER_exponential_MLPNODES_0_t094039/REPETITION_1"
# ]

for idy, model_path in enumerate(models_to_evaluate):
    MODEL_INFO = {
        "pretraining_type": "classification",  # or SimCLR
        "path": model_path,
    }
    DATASET_INFO = {
        "dataset_split_path": os.path.join(
            MODEL_INFO["path"], "data_split_information.csv"
        ),
        "nbr_training_samples_to_embed": None,  # this is an augmented version fo the training set.
    }

    model_version = str(
        os.path.basename(os.path.dirname(pathlib.Path(MODEL_INFO["path"])))
    ).split("_")[0]

    session_time = os.path.basename(
        os.path.dirname(pathlib.Path(MODEL_INFO["path"]))
    ).split("_")[-1]
    repetition = str(os.path.basename(pathlib.Path(MODEL_INFO["path"])))

    SAVE_PATH = pathlib.Path(
        os.path.join(MODEL_INFO["path"], "Feature_classification_experiments")
    )
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

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

        img_mean = [0.5] * 3
        img_std = [0.5] * 3
        preprocess = T.Compose(
            [
                T.Resize(size=(224, 224), antialias=True),
                T.ToTensor(),
                T.Normalize(
                    mean=img_mean,
                    std=img_std,
                ),
            ],
        )
        transforms = T.Compose(
            [
                T.RandomResizedCrop(
                    size=(224, 224),
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
                    mean=img_mean,
                    std=img_std,
                ),
            ],
        )

        # build adataloaders
        training_dataset = DataLoader(
            dataset_utilities.PNGDatasetFromFolder(
                list(training_files_for_inference["file_path"]),
                transform=preprocess,
                labels=training_labels,
                return_file_path=False,
            ),
            num_workers=0,
            batch_size=32,
        )

        validation_dataset = DataLoader(
            dataset_utilities.PNGDatasetFromFolder(
                list(validation_files_for_inference["file_path"]),
                transform=preprocess,
                labels=validation_labels,
                return_file_path=False,
            ),
            num_workers=0,
            batch_size=32,
        )

        # %% LOAD THIS FOLD"S MODEL
        model_path = os.path.join(fold, "last.pt")
        print(model_path)
        if MODEL_INFO["pretraining_type"] == "classification":
            original_model = torch.load(model_path)
            original_model = deepcopy(original_model.model.model)
            feature_extractor = deepcopy(original_model)
        elif MODEL_INFO["pretraining_type"] == "SimCLR":
            original_model = (
                model_bucket_CBTN_v1.SimCLRModelWrapper.load_from_checkpoint(
                    model_path,
                )
            )
            original_model = deepcopy(original_model.convnet)
            feature_extractor = deepcopy(original_model)

        # remove classification head to only get the features
        if "resnet" in model_version.lower():
            if model_version.lower() == "resnet9":
                feature_extractor.classifier[2] = torch.nn.Identity()
            else:
                feature_extractor.fc = torch.nn.Identity()
        elif "vit" in model_version.lower():
            feature_extractor.heads = torch.nn.Identity()

        original_model.to(device)
        original_model.eval()
        feature_extractor.to(device)
        feature_extractor.eval()

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
        original_DL_classifier_performance = {"training": {}, "validation": {}}

        for data, data_name in zip(
            (training_dataset, validation_dataset), ("training", "validation")
        ):
            pred, labels = get_original_model_prediction(original_model, data, device)
            original_DL_classifier_performance[data_name][
                f"original_DL_accuracy"
            ] = accuracy_score(
                np.argmax(labels, axis=1), np.argmax(softmax(pred, axis=1), axis=1)
            )
            original_DL_classifier_performance[data_name][
                f"original_DL_mcc"
            ] = matthews_corrcoef(
                np.argmax(labels, axis=1), np.argmax(softmax(pred, axis=1), axis=1)
            )

            # save confusion matrix
            evaluation_utilities.plotConfusionMatrix(
                GT=labels,
                PRED=softmax(pred, axis=1),
                classes=list(unique_targe_classes.keys()),
                savePath=SAVE_PATH,
                saveName=f"Original_model_{data_name}_performance_fold_{fold_idx}",
                draw=False,
            )

            # ## USE TORCH
            import torchmetrics

            confusion_matrix = torchmetrics.ConfusionMatrix(
                task="multiclass",
                num_classes=len(list(unique_targe_classes.keys())),
                threshold=0.05,
            )

            confusion_matrix(
                torch.argmax(torch.tensor(softmax(pred, axis=1)), dim=1),
                torch.argmax(torch.tensor(labels), dim=1),
            )
            confusion_matrix_computed = (
                confusion_matrix.compute().detach().cpu().numpy().astype(int)
            )

            string_labels = np.unique(list(unique_targe_classes.keys()))
            df_cm = pd.DataFrame(
                confusion_matrix_computed, index=string_labels, columns=string_labels
            )
            fig_, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
            sns.heatmap(
                df_cm, annot=True, cmap="Blues", annot_kws={"size": 20}, ax=ax, fmt="g"
            )
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            fig_.savefig(
                os.path.join(
                    SAVE_PATH,
                    f"Original_model_{data_name}_performance_fold_{fold_idx}_TORCH.png",
                ),
                bbox_inches="tight",
                dpi=100,
            )
            plt.close(fig_)

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
        training_features, training_labels = get_features(
            feature_extractor,
            training_dataset,
            nbr_images_to_generate=DATASET_INFO["nbr_training_samples_to_embed"],
        )

        validation_freatures, validation_labels = get_features(
            feature_extractor,
            validation_dataset,
            nbr_images_to_generate=None,
        )

        # %% APPLY PCA ON THE TRAINING SET AND, USING THE SAME TRANSFORMATION, ON THE VALIDATION SET
        # work on plotting first (always PCA components = 3)
        pca = PCA(n_components=3)
        pca_training = pca.fit_transform(training_features)
        pca_validation = pca.transform(validation_freatures)

        # plot first 3 dimensions
        plot_PCA_embeddings(
            pca_training,
            hue_labels=one_hot_to_class_string(
                training_labels, target_class_to_one_hot_mapping
            ),
            style_labels=list(training_files_for_inference["subject_IDs"]),
            draw=False,
            save_figure=True,
            save_path=SAVE_PATH,
            prefix=f"PCA_training_set_fold_{fold_idx}",
            nbr_legend_columns=4,
        )

        plot_PCA_embeddings(
            pca_validation,
            hue_labels=one_hot_to_class_string(
                validation_labels, target_class_to_one_hot_mapping
            ),
            style_labels=list(validation_files_for_inference["subject_IDs"]),
            draw=False,
            save_figure=True,
            save_path=SAVE_PATH,
            prefix=f"PCA_validation_set_fold_{fold_idx}",
        )

        # reperform PCA with the set number of components
        pca = PCA(n_components=PCA_NBR_COMPONENTS)
        pca_training = pca.fit_transform(training_features)
        pca_validation = pca.transform(validation_freatures)

        # %% FOR ALL THE CLASSIFIERS FIT, AND GET STATS
        from matplotlib.colors import ListedColormap

        # preprocess dataset, split into training and test part
        X_train, X_validation, y_train, y_validation = (
            pca_training,
            pca_validation,
            np.argmax(training_labels, axis=1),
            np.argmax(validation_labels, axis=1),
        )

        # countour plot only works well if PCA components is == 2
        if PCA_NBR_COMPONENTS == 2:
            xx, yy = make_meshgrid(X_train[:, 0], X_train[:, 1], h=0.5)

            # just plot the dataset first
            cm_decision_boundary = plt.cm.Pastel1

            fig, axis = plt.subplots(
                nrows=2, ncols=len(classifiers) + 1, figsize=(25, 5)
            )
            # fix axes limits
            for ax in axis.ravel():
                ax.set_xlim(xx.min(), xx.max())
                ax.set_ylim(yy.min(), yy.max())
                ax.set_xticks(())
                ax.set_yticks(())

            # Plot the training points
            sns.scatterplot(
                x=X_train[:, 0],
                y=X_train[:, 1],
                hue=y_train,
                legend=False,
                ax=axis[0, 0],
            )
            axis[0, 0].set_title(f"Training (PCA dim1 vs dim 2)", fontsize=10)

            # and validation points
            sns.scatterplot(
                x=X_validation[:, 0],
                y=X_validation[:, 1],
                hue=y_validation,
                legend=False,
                ax=axis[1, 0],
            )
            axis[1, 0].set_title(f"Validation (PCA dim1 vs dim 2)", fontsize=10)

        # iterate over classifiers
        performance = {"training": {}, "validation": {}}
        for idc, (name, clf) in enumerate(zip(names, classifiers)):
            # fit the classifier to the training set
            clf.fit(X_train, y_train)
            # get plots and scores for the training and the valdiation
            for idz, (set_name, X, y) in enumerate(
                zip(
                    ("training", "validation"),
                    (X_train, X_validation),
                    (y_train, y_validation),
                )
            ):
                print(f"Working on {name} classifier ({set_name}).")

                if PCA_NBR_COMPONENTS == 2:
                    plot_contours(
                        axis[idz, idc + 1],
                        clf,
                        xx,
                        yy,
                        cmap=cm_decision_boundary,
                        alpha=0.8,
                    )

                    # Plot
                    sns.scatterplot(
                        x=X[:, 0],
                        y=X[:, 1],
                        hue=y,
                        legend=False,
                        ax=axis[idz, idc + 1],
                    )

                # save information for this classifier ad this set
                pred = clf.predict(X)
                performance[set_name][f"{name}_accuracy"] = accuracy_score(y, pred)
                performance[set_name][f"{name}_mcc"] = matthews_corrcoef(y, pred)

                if PCA_NBR_COMPONENTS == 2:
                    # add title
                    axis[idz, idc + 1].set_title(
                        f"{name} (MCC: {performance[set_name][f'{name}_mcc']:0.2f})",
                        fontsize=5,
                    )
        if PCA_NBR_COMPONENTS == 2:
            fig.savefig(
                os.path.join(SAVE_PATH, f"Classifiers performance_{fold_idx}.pdf"),
                dpi=100,
                bbox_inches="tight",
            )
            plt.close(fig)

        # %% SAVE PERFORMANCE SUMMARY FOR THIS FOLD
        for set_name in ["training", "validation"]:
            # basinc information
            row = [
                f"{model_version}_{session_time}",
                repetition,
                len(np.unique(y_train)),
                "_".join(list(unique_targe_classes.keys())),
                "full" if X_train.shape[0] > 1500 else "[45,55]",
                fold_idx,
                set_name,
                PCA_NBR_COMPONENTS,
            ]
            # add original DL classification performance
            row.extend(
                [v for v in original_DL_classifier_performance[set_name].values()]
            )
            # add the feature classifiers
            row.extend([v for v in performance[set_name].values()])
            # append to the summary
            summary_performance.append(row)
    # %% SAVE PERFORMANCE TO CSV FILE

    # columns in the csv file
    columns = [
        "model_version",
        "repetition",
        "nbr_classes",
        "classes",
        "dataset_version",
        "fold_nbr",
        "set",
        "nbr_PCA_components",
    ]
    # add the remaining columns
    columns.extend([k for k in original_DL_classifier_performance["validation"].keys()])
    columns.extend([k for k in performance["validation"].keys()])

    df = pd.DataFrame(data=summary_performance, columns=columns)
    df.to_csv(
        os.path.join(
            SAVE_PATH, f"summary_performance_{PCA_NBR_COMPONENTS}_PCA_components.csv"
        ),
        index=False,
    )

# %%
