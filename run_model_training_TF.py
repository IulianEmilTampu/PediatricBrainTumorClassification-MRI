# %%
"""
Main script that runs model training for tumor detection (binary classification of if a 2D transversal image contains or not tumor) 
in the context of the paediatric brian tumor project.

Steps
1 - get the path to the dataset
2 - build data generator and model
3 - run training routine for the classification of the slices
4 - save model
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import glob
import csv
import json
import numpy as np
import argparse
import importlib
import logging
import random
from pathlib import Path
from distutils.util import strtobool

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils.extmath import softmax

# local imports
import utilities
import data_utilities
import models
import losses
import tf_callbacks

su_debug_flag = True

# --------------------------------------
# read the input arguments and set the base folder
# --------------------------------------
if not su_debug_flag:
    parser = argparse.ArgumentParser(
        description="Run cross validation training for tumor detection."
    )
    parser.add_argument(
        "-wf",
        "--WORKING_FOLDER",
        required=True,
        type=str,
        help="Provide the working folder where the trained model will be saved.",
    )
    parser.add_argument(
        "-df",
        "--IMG_DATASET_FOLDER",
        required=True,
        type=str,
        help="Provide the Image Dataset Folder where the folders for each modality are located (see dataset specifications in the README file).",
    )
    parser.add_argument(
        "-dt",
        "--DATASET_TYPE",
        required=False,
        default="CBTN",
        type=str,
        help="Provide the image dataset type (BRATS, CBTN, CUSTOM). This will set the dataloader appropriate for the dataset.",
    )
    parser.add_argument(
        "-gpu",
        "--GPU_NBR",
        default=0,
        type=str,
        help="Provide the GPU number to use for training.",
    )
    parser.add_argument(
        "-model_name",
        "--MODEL_NAME",
        required=False,
        type=str,
        default="MICCAI2023",
        help="Name used to save the model and the scores",
    )
    parser.add_argument(
        "-n_classes",
        "--NBR_CLASSES",
        required=False,
        type=int,
        default=2,
        help="Number of classification classes.",
    )
    parser.add_argument(
        "-n_folds",
        "--NBR_FOLDS",
        required=False,
        type=int,
        default=1,
        help="Number of cross validation folds.",
    )
    parser.add_argument(
        "-lr",
        "--LEARNING_RATE",
        required=False,
        type=float,
        default=0.0001,
        help="Learning rate",
    )
    parser.add_argument(
        "-batch_size",
        "--BATCH_SIZE",
        required=False,
        type=int,
        default=16,
        help="Specify batch size. Default 16",
    )
    parser.add_argument(
        "-e",
        "--MAX_EPOCHS",
        required=False,
        type=int,
        default=300,
        help="Number of max training epochs.",
    )
    parser.add_argument(
        "-use_pretrained",
        "--USE_PRETRAINED_MODEL",
        required=False,
        dest="USE_PRETRAINED_MODEL",
        type=lambda x: bool(strtobool(x)),
        help="Specify if the image encoder should be loading the weight pretrained on BraTS",
    )
    parser.add_argument(
        "-path_to_pretrained_model",
        "--PATH_TO_PRETRAINED_MODEL",
        required=False,
        type=str,
        default=None,
        help="Specify the path to the pretrained model to use as image encoder.",
    )
    parser.add_argument(
        "-use_age",
        "--USE_AGE",
        required=False,
        dest="USE_AGE",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Specify if the model should use the age information. If true, the age information is encoded using a fuly connected model and feature fusion is used to combine image and age infromation.",
    )
    parser.add_argument(
        "-use_gradCAM",
        "--USE_GRADCAM",
        required=False,
        dest="USE_GRADCAM",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Specify if the model should use the gradCAM information. If true, the gradCAM infromation is concatenated to the image information as an extra channel",
    )
    parser.add_argument(
        "-loss",
        "--LOSS",
        required=False,
        type=str,
        default="CCE",
        help="Specify loss to use during model training (categorical cross entropy CCE, MCC, binary categorical cross entropy BCE. Other can be defined and used. Just implement.",
    )
    parser.add_argument(
        "-mr_modelities",
        "--MR_MODALITIES",
        nargs="+",
        required=False,
        default=["T2"],
        help="Specify which MR modalities to use during training (T1 and/or T2)",
    )
    parser.add_argument(
        "-debug_dataset_fraction",
        "--DEBUG_DATASET_FRACTION",
        required=False,
        type=float,
        default=1.0,
        help="Specify the percentage of the dataset to use during training and validation. This is for debug",
    )
    parser.add_argument(
        "-trf_data",
        "--TFR_DATA",
        required=False,
        dest="TFR_DATA",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Specify if the dataset used for training originates from TFRecord files. This is used to choose between data generators.",
    )
    parser.add_argument(
        "-model_type",
        "--MODEL_TYPE",
        required=False,
        type=str,
        default="SDM4",
        help="Specify model to use during training. Chose among the ones available in the models.py file.",
    )
    parser.add_argument(
        "-optimizer",
        "--OPTIMIZER",
        required=False,
        type=str,
        default="SGD",
        help="Specify which optimizer to use. Here one can set SGD or ADAM. Others can be implemented.",
    )
    # other parameters
    parser.add_argument(
        "-rns",
        "--RANDOM_SEED_NUMBER",
        required=False,
        type=int,
        default=29122009,
        help="Specify random number seed. Useful to have models trained and tested on the same data.",
    )

    args_dict = dict(vars(parser.parse_args()))

else:
    # # # # # # # # # # # # # # DEBUG
    args_dict = {
        "WORKING_FOLDER": "/flush/iulta54/Research/P5-MICCAI2023",
        "IMG_DATASET_FOLDER": "/flush/iulta54/Research/Data/CBTN/EXTRACTED_SLICES_TFR",
        "DATASET_TYPE": "CBTN",
        "NBR_CLASSES": 3,
        "GPU_NBR": "0",
        "NBR_FOLDS": 1,
        "LEARNING_RATE": 0.00001,
        "BATCH_SIZE": 32,
        "MAX_EPOCHS": 50,
        "USE_PRETRAINED_MODEL": True,
        "PATH_TO_PRETRAINED_MODEL": "/flush/iulta54/Research/P5-MICCAI2023/trained_models_archive/SDM4_t2_BraTS_fullDataset_lr10em6_more_data/fold_1/last_model",
        "USE_AGE": False,
        "USE_GRADCAM": False,
        "LOSS": "MCC_and_CCE_Loss",
        "RANDOM_SEED_NUMBER": 1214,
        "MR_MODALITIES": ["T2"],
        "DEBUG_DATASET_FRACTION": 0.6,
        "TFR_DATA": True,
        "MODEL_TYPE": "EfficientNet",
        "MODEL_NAME": "TEST_model_capacity_EfficientNet_preTrained",
        "OPTIMIZER": "ADAM",
    }

# revise model name
args_dict[
    "MODEL_NAME"
] = f'{args_dict["MODEL_NAME"]}_optm_{args_dict["OPTIMIZER"]}_{args_dict["MODEL_TYPE"]}_TFRdata_{args_dict["TFR_DATA"]}_modality_{"_".join([i for i in args_dict["MR_MODALITIES"]])}_loss_{args_dict["LOSS"]}_lr_{args_dict["LEARNING_RATE"]}_batchSize_{args_dict["BATCH_SIZE"]}_pretrained_{args_dict["USE_PRETRAINED_MODEL"]}_useAge_{args_dict["USE_AGE"]}_useGradCAM_{args_dict["USE_GRADCAM"]}'

# --------------------------------------
# set GPU (or device)
# --------------------------------------

# import tensorflow
import tensorflow as tf
import tensorflow_addons as tfa
import warnings

tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.utils import to_categorical
from tensorflow_addons.optimizers import Lookahead

# from tensorflow.python.framework.ops import disable_eager_execution

# disable_eager_execution()

devices = tf.config.list_physical_devices("GPU")

if devices:
    print(f'Running training on GPU # {args_dict["GPU_NBR"]} \n')
    warnings.simplefilter(action="ignore", category=FutureWarning)
    tf.config.experimental.set_memory_growth(devices[0], True)
else:
    Warning(
        f"ATTENTION!!! MODEL RUNNING ON CPU. Check implementation in case GPU is wanted."
    )
# -------------------------------------
# Check that the given folder exist
# -------------------------------------
for folder, fd in zip(
    [
        args_dict["WORKING_FOLDER"],
        args_dict["IMG_DATASET_FOLDER"],
    ],
    [
        "working folder",
        "image dataset folder",
    ],
):
    if not os.path.isdir(folder):
        raise ValueError(f"{fd.capitalize} not found. Given {folder}.")

# -------------------------------------
# Create folder where to save the model
# -------------------------------------
args_dict["SAVE_PATH"] = os.path.join(
    args_dict["WORKING_FOLDER"], "trained_models_archive", args_dict["MODEL_NAME"]
)
Path(args_dict["SAVE_PATH"]).mkdir(parents=True, exist_ok=True)

# print input variables
max_len = max([len(key) for key in args_dict])
[
    print(f"{key:{max_len}s}: {value} ({type(value)})")
    for key, value in args_dict.items()
]
# %% GET DATASET FILES
print(f"Splitting dataset (per-volume (subject) splitting).")
"""
Independently from the combination of modalities, the test validation and train sets
are defined so that no vlomume is present in more than one set.

Steps
2 - using the number of subject, screate indexes to identify which files
    are used for training, validation and testing
3 - save the information about the split.
"""
importlib.reload(data_utilities)

all_file_names = data_utilities.get_img_file_names(
    img_dataset_path=args_dict["IMG_DATASET_FOLDER"],
    dataset_type="CBTN",
    modalities=args_dict["MR_MODALITIES"],
    return_labels=True,
    task="detection" if args_dict["NBR_CLASSES"] == 2 else "classification",
    nbr_classes=args_dict["NBR_CLASSES"],
    tumor_min_rpl=0,
    tumor_max_rpl=100,
    brain_min_rpl=1,
    brain_max_rpl=100,
    file_format="tfrecords",
    tumor_loc=["infra", "supra"],
)

unique_patien_IDs_with_labels = list(
    dict.fromkeys([(f[1], f[2]) for f in all_file_names])
)
unique_patien_IDs_labels = [f[1] for f in unique_patien_IDs_with_labels]

args_dict["NBR_SUBJECTS"] = len(unique_patien_IDs_with_labels)

subj_train_val_idx, subj_test_idx = train_test_split(
    unique_patien_IDs_with_labels,
    stratify=unique_patien_IDs_labels,
    test_size=0.15,
    random_state=args_dict["RANDOM_SEED_NUMBER"],
)
test_files = [
    f[0] for f in all_file_names if any([i[0] == f[1] for i in subj_test_idx])
]
print(f'{"# Train-val subjects":18s}: {len(subj_train_val_idx):2d}')
print(
    f'{"# Test subjects":18s}: {len(subj_test_idx):2d} ({subj_test_idx} {len(test_files)} total images)'
)
# get labels for the remaining files so that can perform stratified cross validation
subj_train_val_idx_labels = [f[1] for f in subj_train_val_idx]
class_weights_values = list(
    np.sum(np.bincount(subj_train_val_idx_labels))
    / (
        len(np.bincount(subj_train_val_idx_labels))
        * np.bincount(subj_train_val_idx_labels)
    )
)

args_dict["CLASS_WEIGHTS"] = {}

for c in range(args_dict["NBR_CLASSES"]):
    args_dict["CLASS_WEIGHTS"][c] = class_weights_values[c] ** 2

subj_train_idx, subj_val_idx = [], []
per_fold_training_files, per_fold_validation_files = [], []
# set cross validation
if args_dict["NBR_FOLDS"] > 1:
    kf = KFold(
        n_splits=args_dict["NBR_FOLDS"],
        shuffle=True,
        random_state=args_dict["RANDOM_SEED_NUMBER"],
    )
    for idx, (train_index, val_index) in enumerate(
        kf.split(subj_train_val_idx, subj_train_val_idx_labels)
    ):
        subj_train_idx.append([subj_train_val_idx[i] for i in train_index])
        subj_val_idx.append([subj_train_val_idx[i] for i in val_index])
        # get also the respective training and validation file names for the generator
        per_fold_training_files.append(
            [
                f[0]
                for f in all_file_names
                if any([i[0] == f[1] for i in subj_train_idx[-1]])
            ]
        )
        per_fold_validation_files.append(
            [
                f[0]
                for f in all_file_names
                if any([i[0] == f[1] for i in subj_val_idx[-1]])
            ]
        )

        # print to check that all is good
        print(
            f'Fold {idx+1}: \n {""*4}{"training":10s} ->{subj_train_idx[-1]} \n {""*4}{"validation":10s} ->{subj_val_idx[-1]}'
        )
else:
    # N_FOLDS is only one, setting 10% of the training dataset as validation
    print("Getting indexes of training and validation files for one fold.")
    aus_train, aus_val = train_test_split(
        subj_train_val_idx,
        stratify=subj_train_val_idx_labels,
        test_size=0.1,
        random_state=args_dict["RANDOM_SEED_NUMBER"],
    )
    subj_train_idx.append(aus_train)
    subj_val_idx.append(aus_val)
    per_fold_training_files.append(
        [
            f[0]
            for f in all_file_names
            if any([i[0] == f[1] for i in subj_train_idx[-1]])
        ]
    )
    per_fold_validation_files.append(
        [f[0] for f in all_file_names if any([i[0] == f[1] for i in subj_val_idx[-1]])]
    )

    # print to check that all is good
    print(
        f'Fold {args_dict["NBR_FOLDS"]}: \n {""*4}{"training":10s} ->{subj_train_idx[-1]} ({len(per_fold_training_files[-1])} images) \n {""*4}{"validation":10s} ->{subj_val_idx[-1]} ({len(per_fold_validation_files[-1])} images)'
    )

# check that no testing files are in the training or validation
idx_to_remove = []
remove_overlap = True
for idx, test_f in enumerate(test_files):
    print(
        f"Checking test files ({idx+1:0{len(str(len(test_files)))}d}\{len(test_files)})\r",
        end="",
    )
    # check in each fold
    for fold in range(len(per_fold_training_files)):
        # check in the training set
        if any([test_f == f for f in per_fold_training_files[fold]]):
            if remove_overlap:
                idx_to_remove.append(idx)
            else:
                raise ValueError(
                    f"ATTENTION!!!\nSome of the testing files are part of the training set!\nCheck implementation"
                )
        if any([test_f == f for f in per_fold_validation_files[fold]]):
            if remove_overlap:
                idx_to_remove.append(idx)
            else:
                raise ValueError(
                    f"ATTENTION!!!\nSome of the testing files are part of the training set!\nCheck implementation"
                )
if remove_overlap:
    test_f = [f for idx, f in enumerate(test_f) if idx not in idx_to_remove]

print(f"\nChecking of the test files passed!")

# Save infromation about which files are used for training/validation/testing
dict = {
    "test": [os.path.basename(f) for f in test_files],
    "train": [],
    "validation": [],
}

for f in range(args_dict["NBR_FOLDS"]):
    dict["train"].append([os.path.basename(i) for i in per_fold_training_files[f]])
    dict["validation"].append(
        [os.path.basename(i) for i in per_fold_validation_files[f]]
    )

with open(
    os.path.join(args_dict["SAVE_PATH"], "train_val_test_files.json"), "w"
) as file:
    json.dump(dict, file)

print(
    f"Training files:{len(per_fold_training_files[-1])}\nValidation files: {len(per_fold_validation_files[-1])}"
)

# %% test generators

look_at_generator = False
if look_at_generator:
    # define utilities

    def show_batched_example_tfrs_dataset(
        dataset,
        class_names=["0", "1"],
        nbr_images: int = 1,
        show_gradCAM: bool = False,
        show_histogram: bool = False,
    ):
        dataset_iterator = iter(dataset)
        image_batch, label_batch = next(dataset_iterator)
        print(image_batch["image"].shape)
        print(
            f' mean: {np.mean(image_batch["image"]):0.4f}\n std: {np.std(image_batch["image"]):0.4f}'
        )

        for i in range(nbr_images):
            fig, ax = plt.subplots(
                nrows=1, ncols=2 if show_gradCAM else 1, figsize=(5, 5)
            )

            if show_gradCAM:
                ax[0].imshow(image_batch["image"][i, :, :, 0], cmap="gray")
                label = label_batch[i]
                ax[0].set_title(class_names[label.argmax()])
                ax[1].imshow(image_batch["image"][i, :, :, 1], cmap="gray")
            else:
                ax.imshow(image_batch["image"][i, :, :, 0], cmap="gray")
                label = label_batch[i]
                ax.set_title(class_names[label.numpy().argmax()])
            plt.show(fig)

        if show_histogram:
            fig, ax = plt.subplots(
                nrows=1, ncols=2 if show_gradCAM else 1, figsize=(5, 5)
            )
            if show_gradCAM:
                ax[0].hist(
                    image_batch["image"][:, :, :, 0].numpy().ravel(),
                )
                label = label_batch[i]
                ax[0].set_title("Histogram of image pixel values")
                ax[1].hist(
                    image_batch["image"][:, :, :, 1].numpy().ravel(),
                )
            else:
                ax.hist(
                    image_batch["image"][:, :, :, 0].numpy().ravel(),
                )
                label = label_batch[i]
                ax.set_title("Histogram of image pixel values")

    def show_batched_example(dataset, class_names=["0", "1"], nbr_images: int = 1):
        image_batch, label_batch = next(iter(dataset))

        plt.figure(figsize=(10, 10))
        for i in range(nbr_images):
            plt.imshow(image_batch["image"][i, :, :, 1], cmap="gray")
            label = label_batch.numpy()[i]
            plt.title(class_names[label.numpy().argmax()])
            plt.axis("off")

    importlib.reload(data_utilities)
    target_size = (224, 224)

    random.shuffle(test_files)

    gen, gen_steps = data_utilities.tfrs_data_generator(
        file_paths=test_files,
        input_size=target_size,
        batch_size=1,
        buffer_size=1000,
        data_augmentation=True,
        normalize_img=False,
        return_age=True,
        normalize_age=False,
        return_gradCAM=True,
        normalize_gradCAM=False,
        dataset_type="test",
        nbr_classes=args_dict["NBR_CLASSES"],
    )

    img_stats, gradCAM_stats, age_stats = data_utilities.get_normalization_values(
        gen, gen_steps, return_age_norm_values=True, return_gradCAM_norm_values=True
    )

    print(f"Image data (mean+-std): {img_stats}")
    print(f"gradCAM data (mean+-std): {gradCAM_stats}")
    print(f"Age data (mean+-std): {age_stats}")

    gen, gen_steps = data_utilities.tfrs_data_generator(
        file_paths=test_files,
        input_size=target_size,
        batch_size=20,
        buffer_size=1000,
        data_augmentation=True,
        normalize_img=True,
        img_norm_values=img_stats,
        return_gradCAM=True,
        normalize_gradCAM=True,
        gradCAM_norm_values=None,
        return_age=True,
        normalize_age=True,
        age_norm_values=age_stats,
        dataset_type="train",
        nbr_classes=args_dict["NBR_CLASSES"],
    )

    img_stats, gradCAM_stats, age_stats = data_utilities.get_normalization_values(
        gen, gen_steps, return_age_norm_values=True, return_gradCAM_norm_values=True
    )

    print("################## AFTER NORMALIZATION #################")
    print(f"Image data (mean+-std): {img_stats}")
    print(f"gradCAM data (mean+-std): {gradCAM_stats}")
    print(f"Age data (mean+-std): {age_stats}")

    if Path(test_files[0]).suffix == ".tfrecords":
        show_batched_example_tfrs_dataset(
            gen,
            nbr_images=15,
            show_gradCAM=False,
            show_histogram=True,
            class_names=["ASTR", "EP", "MED"],
        )
    else:
        show_batched_example(gen, nbr_images=5)

# %%
# ---------
# RUNIING CROSS VALIDATION TRAINING
# ---------
importlib.reload(data_utilities)
importlib.reload(models)

# save training configuration (right before training to account for the changes made in the meantime)
with open(os.path.join(args_dict["SAVE_PATH"], "config.json"), "w") as config_file:
    config_file.write(json.dumps(args_dict))

# create dictionary where to save the test performance
summary_test = {}

for cv_f in range(args_dict["NBR_FOLDS"]):
    # make forder where to save the model
    save_model_path = os.path.join(args_dict["SAVE_PATH"], f"fold_{cv_f+1}")
    Path(save_model_path).mkdir(parents=True, exist_ok=True)
    summary_test[str(cv_f + 1)] = {"best": [], "last": []}

    print(f'{" "*3}Setting up training and validation data Generators ...')

    # --------------------------
    # CREATE DATA GENERATORS
    # -------------------------

    # specify data generator specific for the different types of datasets
    if args_dict["TFR_DATA"]:
        data_gen = data_utilities.tfrs_data_generator
    elif any(
        [args_dict["USE_AGE"], args_dict["USA_GRADCAM"], not args_dict["TFR_DATA"]]
    ):
        raise ValueError(
            "Trying to run training using dataset from .png files while asking for age information and/or gradCAM.\nThis is not yet implemented. Use TFR dataset."
        )
    else:
        data_gen = data_utilities.img_data_generator

    # define generator parameters
    target_size = (224, 224)

    # # ################## TRAINING GENERATOR (get also normalization values)
    tr_files = per_fold_training_files[cv_f]
    random.Random(args_dict["RANDOM_SEED_NUMBER"]).shuffle(tr_files)

    if not all(
        [args_dict["USE_PRETRAINED_MODEL"], args_dict["MODEL_TYPE"] == "EfficientNet"]
    ):
        flag_normalization = True
        train_gen, train_steps = data_gen(
            file_paths=tr_files[
                0 : int(len(tr_files) * args_dict["DEBUG_DATASET_FRACTION"])
            ],
            input_size=target_size,
            batch_size=args_dict["BATCH_SIZE"],
            buffer_size=1000,
            data_augmentation=False,
            normalize_img=False,
            img_norm_values=None,
            return_gradCAM=args_dict["USE_GRADCAM"],
            normalize_gradCAM=False,
            gradCAM_norm_values=None,
            return_age=args_dict["USE_AGE"],
            normalize_age=False,
            age_norm_values=None,
            dataset_type="test",
            nbr_classes=args_dict["NBR_CLASSES"],
        )

        # get normalization stats
        print(f'{" "*6}Getting normalization stats from training generator...')
        norm_stats = data_utilities.get_normalization_values(
            train_gen,
            train_steps,
            return_age_norm_values=args_dict["USE_AGE"],
            return_gradCAM_norm_values=args_dict["USE_GRADCAM"],
        )
    else:
        flag_normalization = False
        norm_stats = None

    # build actuall training datagen with normalized values
    train_gen, train_steps = data_gen(
        file_paths=tr_files[
            0 : int(len(tr_files) * args_dict["DEBUG_DATASET_FRACTION"])
        ],
        input_size=target_size,
        batch_size=args_dict["BATCH_SIZE"],
        buffer_size=1000,
        data_augmentation=False,
        normalize_img=True if flag_normalization else False,
        img_norm_values=norm_stats[0]
        if any([args_dict["USE_GRADCAM"], args_dict["USE_AGE"]])
        else norm_stats,
        return_gradCAM=args_dict["USE_GRADCAM"],
        normalize_gradCAM=True,
        gradCAM_norm_values=norm_stats[1] if args_dict["USE_GRADCAM"] else None,
        return_age=args_dict["USE_AGE"],
        normalize_age=True,
        age_norm_values=norm_stats[2]
        if all([args_dict["USE_AGE"], args_dict["USE_GRADCAM"]])
        else norm_stats[1]
        if all([args_dict["USE_AGE"], not args_dict["USE_GRADCAM"]])
        else None,
        dataset_type="train",
        nbr_classes=args_dict["NBR_CLASSES"],
    )
    print(f'{" "*6}Training gen. done!')

    val_files = per_fold_validation_files[cv_f]
    random.Random(args_dict["RANDOM_SEED_NUMBER"]).shuffle(val_files)
    val_gen, val_steps = data_gen(
        file_paths=val_files[
            0 : int(len(val_files) * args_dict["DEBUG_DATASET_FRACTION"])
        ],
        input_size=target_size,
        batch_size=args_dict["BATCH_SIZE"],
        buffer_size=1000,
        data_augmentation=False,
        normalize_img=True if flag_normalization else False,
        img_norm_values=norm_stats[0]
        if any([args_dict["USE_GRADCAM"], args_dict["USE_AGE"]])
        else norm_stats,
        return_gradCAM=args_dict["USE_GRADCAM"],
        normalize_gradCAM=True,
        gradCAM_norm_values=norm_stats[1] if args_dict["USE_GRADCAM"] else None,
        return_age=args_dict["USE_AGE"],
        normalize_age=True,
        age_norm_values=norm_stats[2]
        if all([args_dict["USE_AGE"], args_dict["USE_GRADCAM"]])
        else norm_stats[1]
        if all([args_dict["USE_AGE"], not args_dict["USE_GRADCAM"]])
        else None,
        dataset_type="val",
        nbr_classes=args_dict["NBR_CLASSES"],
    )
    print(f'{" "*6}Validation gen. done!')
    test_gen, test_steps = data_gen(
        file_paths=test_files,
        input_size=target_size,
        batch_size=args_dict["BATCH_SIZE"],
        buffer_size=10,
        data_augmentation=False,
        normalize_img=True if flag_normalization else False,
        img_norm_values=norm_stats[0]
        if any([args_dict["USE_GRADCAM"], args_dict["USE_AGE"]])
        else norm_stats,
        return_gradCAM=args_dict["USE_GRADCAM"],
        normalize_gradCAM=True,
        gradCAM_norm_values=norm_stats[1] if args_dict["USE_GRADCAM"] else None,
        return_age=args_dict["USE_AGE"],
        normalize_age=True,
        age_norm_values=norm_stats[2]
        if all([args_dict["USE_AGE"], args_dict["USE_GRADCAM"]])
        else norm_stats[1]
        if all([args_dict["USE_AGE"], not args_dict["USE_GRADCAM"]])
        else None,
        dataset_type="test",
        nbr_classes=args_dict["NBR_CLASSES"],
    )
    print(f'{" "*6}Testing gen. done!')

    # -------------
    # CREATE MODEL
    # ------------
    importlib.reload(models)
    if all([args_dict["USE_PRETRAINED_MODEL"], args_dict["MODEL_TYPE"] == "SDM4"]):
        print(f'{" "*3}Loading pretrained model...')
        if args_dict["MODEL_TYPE"] == "SDM4":
            # load model
            model = tf.keras.models.load_model(args_dict["PATH_TO_PRETRAINED_MODEL"])
            # replace the last dense layer to match the number of classes
            intermediat_output = model.layers[-2].output
            new_output = tf.keras.layers.Dense(
                units=len(train_gen.class_indices),
                input_shape=model.layers[-1].input_shape,
                name="prediction",
            )(intermediat_output)
            # make sure that model layers are trainable
            for layer in model.layers:
                layer.trainable = False
            model = tf.keras.Model(inputs=model.inputs, outputs=new_output)
            print(model.summary())
    else:
        print(f'{" "*3}Building model from scratch...')
        # build custom model
        input_shape = (
            (target_size[0], target_size[1], 1)
            if not args_dict["USE_GRADCAM"]
            else (target_size[0], target_size[1], 2)
        )

        if args_dict["MODEL_TYPE"] == "SDM4":
            print(f'{" "*6}Using {args_dict["MODEL_TYPE"]} model.')
            model = models.SimpleDetectionModel_TF(
                num_classes=args_dict["NBR_CLASSES"],
                input_shape=input_shape,
                kernel_size=(3, 3),
                pool_size=(2, 2),
                use_age=args_dict["USE_AGE"],
                use_age_thr_tabular_network=False,
                use_gradCAM=args_dict["USE_GRADCAM"],
            )

        elif args_dict["MODEL_TYPE"] == "ResNet9":
            print(f'{" "*6}Using {args_dict["MODEL_TYPE"]} model.')
            model = models.ResNet9(
                num_classes=args_dict["NBR_CLASSES"],
                input_shape=input_shape,
                use_age=args_dict["USE_AGE"],
                use_age_thr_tabular_network=False,
            )
        elif args_dict["MODEL_TYPE"] == "ViT":
            print(f'{" "*6}Using {args_dict["MODEL_TYPE"]} model.')
            model = models.ViT(
                input_size=input_shape,
                num_classes=args_dict["NBR_CLASSES"],
                use_age=args_dict["USE_AGE"],
                use_age_thr_tabular_network=False,
                use_gradCAM=args_dict["USE_GRADCAM"],
                patch_size=16,
                projection_dim=64,
                num_heads=8,
                mlp_head_units=(256, 128),
                transformer_layers=8,
                transformer_units=None,
                debug=False,
            )
        elif args_dict["MODEL_TYPE"] == "EfficientNet":
            print(f'{" "*6}Using {args_dict["MODEL_TYPE"]} model.')
            model = models.EfficientNet(
                num_classes=args_dict["NBR_CLASSES"],
                input_shape=input_shape,
                use_age=args_dict["USE_AGE"],
                use_age_thr_tabular_network=False,
                pretrained=args_dict["USE_PRETRAINED_MODEL"],
                froze_weights=True,
            )
        else:
            raise ValueError(
                "Model type not among the ones that are implemented.\nDefine model in the models.py file and add code here for building the model."
            )

        # ################################# COMPILE MODEL
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            args_dict["LEARNING_RATE"], args_dict["MAX_EPOCHS"], 0, power=0.99
        )

        if args_dict["OPTIMIZER"] == "SGD":
            print(f'{" "*6}Using SGD optimizer.')
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=args_dict["LEARNING_RATE"],
                decay=1e-6,
                momentum=0.9,
                nesterov=True,
                clipvalue=0.5,
            )
            # optimizer = tf.keras.optimizers.SGD(
            #     learning_rate=learning_rate_fn,
            #     nesterov=True,
            # )
        elif args_dict["OPTIMIZER"] == "ADAM":
            print(f'{" "*6}Using AdamW optimizer.')

            optimizer = tfa.optimizers.AdamW(
                learning_rate=args_dict["LEARNING_RATE"], weight_decay=0.0001
            )
            # optimizer = tfa.optimizers.AdamW(
            #     learning_rate=learning_rate_fn, weight_decay=0.0001
            # )

    # wrap using LookAhead which helps smoothing out validation curves
    optimizer = Lookahead(optimizer, sync_period=5, slow_step_size=0.7)

    if args_dict["LOSS"] == "MCC":
        print(f'{" "*6}Using MCC loss.')
        importlib.reload(losses)
        loss = losses.MCC_Loss()
        what_to_monitor = tfa.metrics.MatthewsCorrelationCoefficient(
            num_classes=args_dict["NBR_CLASSES"]
        )
    elif args_dict["LOSS"] == "MCC_and_CCE_Loss":
        print(f'{" "*6}Using sum of MCC and CCE loss.')
        importlib.reload(losses)
        loss = losses.MCC_and_CCE_Loss()
        what_to_monitor = tfa.metrics.MatthewsCorrelationCoefficient(
            num_classes=args_dict["NBR_CLASSES"]
        )
    elif args_dict["LOSS"] == "CCE":
        print(f'{" "*6}Using CCE loss.')
        loss = tf.keras.losses.CategoricalCrossentropy()
        what_to_monitor = "val_accuracy"
    elif args_dict["LOSS"] == "BCE":
        print(f'{" "*6}Using BCS loss.')
        loss = tf.keras.losses.BinaryCrossentropy()
        what_to_monitor = "val_accuracy"
    else:
        raise ValueError(
            f"The loss provided is not available. Implement in the losses.py or here."
        )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            "accuracy",
            tfa.metrics.MatthewsCorrelationCoefficient(
                num_classes=args_dict["NBR_CLASSES"]
            ),
        ],
    )

    # print model architecture
    tf.keras.utils.plot_model(
        model,
        to_file=os.path.join(save_model_path, "model_architecture.jpeg"),
        show_shapes=True,
        show_layer_activations=True,
        expand_nested=True,
    )

    # ######################### SET MODEL CHECKPOINT
    best_model_path = os.path.join(save_model_path, "best_model_weights", "")
    Path(best_model_path).mkdir(parents=True, exist_ok=True)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

    importlib.reload(tf_callbacks)
    callbacks_list = [
        # tf_callbacks.SaveBestModelWeights(
        #     save_path=best_model_path, monitor="val_loss", mode="min"
        # ),
        model_checkpoint_callback,
        tf_callbacks.LossAndErrorPrintingCallback(
            save_path=save_model_path, print_every_n_epoch=5
        ),
    ]

    # ------------------
    # RUN MODEL TRAINING
    # ------------------
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=args_dict["MAX_EPOCHS"],
        verbose=1,
        callbacks=callbacks_list,
        class_weight=args_dict["CLASS_WEIGHTS"],
    )

    # save last model
    last_model_path = os.path.join(save_model_path, "last_model")
    Path(last_model_path).mkdir(parents=True, exist_ok=True)
    model.save(
        os.path.join(last_model_path, "last_model"),
        save_format="h5",
        include_optimizer=False,
    )
    # ------------------
    # MODEL EVALUATION
    # ------------------
    importlib.reload(utilities)
    print(f'{" "*6}Evaluationg Last model on testing data.')
    # ###################### LAST MODEL
    # get the per_slice classification
    Ptest_softmax = []
    Ytest_categorical = []
    ds_iter = iter(test_gen)
    ds_steps = test_steps
    for i in range(ds_steps):
        x, y = next(ds_iter)
        Ytest_categorical.append(y)
        Ptest_softmax.append(model.predict(x, verbose=0))
    Ptest_softmax = np.row_stack(Ptest_softmax)

    Ptest = np.argmax(Ptest_softmax, axis=-1)
    Ytest_categorical = np.row_stack(Ytest_categorical)
    summary_test[str(cv_f + 1)]["last"] = utilities.get_performance_metrics(
        Ytest_categorical, Ptest_softmax, average="macro"
    )
    summary_test[str(cv_f + 1)]["last"]["per_case_prediction"] = Ptest

    # SAVE CONFUSION MATRIX; ROC and PR curves
    utilities.plotConfusionMatrix(
        GT=Ytest_categorical,
        PRED=Ptest_softmax,
        classes=["Not_tumor", "Tumor"]
        if args_dict["NBR_CLASSES"] == 2
        else ["ASTR", "EP", "MED"],
        savePath=save_model_path,
        saveName="last_model_CM_test",
        draw=False,
    )
    utilities.plotROC(
        GT=Ytest_categorical,
        PRED=Ptest_softmax,
        classes=["Not_tumor", "Tumor"]
        if args_dict["NBR_CLASSES"] == 2
        else ["ASTR", "EP", "MED"],
        savePath=save_model_path,
        saveName="last_model_ROC_test",
        draw=False,
    )
    utilities.plotPR(
        GT=Ytest_categorical,
        PRED=Ptest_softmax,
        classes=["Not_tumor", "Tumor"]
        if args_dict["NBR_CLASSES"] == 2
        else ["ASTR", "EP", "MED"],
        savePath=save_model_path,
        saveName="last_model_PR_test",
        draw=False,
    )

    # do the same for the validation dataset
    print(f'{" "*6}Evaluationg Last model on validation data.')
    Pval_softmax = []
    Yval_categorical = []
    ds_iter = iter(val_gen)
    ds_steps = val_steps
    for i in range(ds_steps):
        x, y = next(ds_iter)
        Yval_categorical.append(y)
        Pval_softmax.append(model.predict(x, verbose=0))
    Pval_softmax = np.row_stack(Pval_softmax)

    Ptest = np.argmax(Pval_softmax, axis=-1)
    Yval_categorical = np.row_stack(Yval_categorical)

    # SAVE CONFUSION MATRIX; ROC and PR curves
    utilities.plotConfusionMatrix(
        GT=Yval_categorical,
        PRED=Pval_softmax,
        classes=["Not_tumor", "Tumor"]
        if args_dict["NBR_CLASSES"] == 2
        else ["ASTR", "EP", "MED"],
        savePath=save_model_path,
        saveName="last_model_CM_validation",
        draw=False,
    )
    utilities.plotROC(
        GT=Yval_categorical,
        PRED=Pval_softmax,
        classes=["Not_tumor", "Tumor"]
        if args_dict["NBR_CLASSES"] == 2
        else ["ASTR", "EP", "MED"],
        savePath=save_model_path,
        saveName="last_model_ROC_validation",
        draw=False,
    )
    utilities.plotPR(
        GT=Yval_categorical,
        PRED=Pval_softmax,
        classes=["Not_tumor", "Tumor"]
        if args_dict["NBR_CLASSES"] == 2
        else ["ASTR", "EP", "MED"],
        savePath=save_model_path,
        saveName="last_model_PR_validation",
        draw=False,
    )

    # ###################### BEST MODEL
    print(f'{" "*6}Evaluationg Best model on testing data.')
    model.load_weights(best_model_path)
    # get the per_slice classification
    Ptest_softmax = []
    Ytest_categorical = []
    ds_iter = iter(test_gen)
    ds_steps = test_steps
    for i in range(ds_steps):
        x, y = next(ds_iter)
        Ytest_categorical.append(y)
        Ptest_softmax.append(model.predict(x, verbose=0))
    Ptest_softmax = np.row_stack(Ptest_softmax)

    Ptest = np.argmax(Ptest_softmax, axis=-1)

    Ytest_categorical = np.row_stack(Ytest_categorical)
    summary_test[str(cv_f + 1)]["best"] = utilities.get_performance_metrics(
        Ytest_categorical, Ptest_softmax, average="macro"
    )
    summary_test[str(cv_f + 1)]["best"]["per_case_prediction"] = Ptest

    # SAVE CONFUSION MATRIX; ROC and PR curves
    utilities.plotConfusionMatrix(
        GT=Ytest_categorical,
        PRED=Ptest_softmax,
        classes=["Not_tumor", "Tumor"]
        if args_dict["NBR_CLASSES"] == 2
        else ["ASTR", "EP", "MED"],
        savePath=save_model_path,
        saveName="best_model_CM",
        draw=False,
    )
    utilities.plotROC(
        GT=Ytest_categorical,
        PRED=Ptest_softmax,
        classes=["Not_tumor", "Tumor"]
        if args_dict["NBR_CLASSES"] == 2
        else ["ASTR", "EP", "MED"],
        savePath=save_model_path,
        saveName="best_model_ROC",
        draw=False,
    )
    utilities.plotPR(
        GT=Ytest_categorical,
        PRED=Ptest_softmax,
        classes=["Not_tumor", "Tumor"]
        if args_dict["NBR_CLASSES"] == 2
        else ["ASTR", "EP", "MED"],
        savePath=save_model_path,
        saveName="best_model_PR",
        draw=False,
    )

    # do the same for the validation dataset
    print(f'{" "*6}Evaluationg Best model on validation data.')
    Pval_softmax = []
    Yval_categorical = []
    ds_iter = iter(val_gen)
    ds_steps = val_steps
    for i in range(ds_steps):
        x, y = next(ds_iter)
        Yval_categorical.append(y)
        Pval_softmax.append(model.predict(x, verbose=0))
    Pval_softmax = np.row_stack(Pval_softmax)

    Ptest = np.argmax(Pval_softmax, axis=-1)
    Yval_categorical = np.row_stack(Yval_categorical)

    # SAVE CONFUSION MATRIX; ROC and PR curves
    utilities.plotConfusionMatrix(
        GT=Yval_categorical,
        PRED=Pval_softmax,
        classes=["Not_tumor", "Tumor"]
        if args_dict["NBR_CLASSES"] == 2
        else ["ASTR", "EP", "MED"],
        savePath=save_model_path,
        saveName="best_model_CM_validation",
        draw=False,
    )
    utilities.plotROC(
        GT=Yval_categorical,
        PRED=Pval_softmax,
        classes=["Not_tumor", "Tumor"]
        if args_dict["NBR_CLASSES"] == 2
        else ["ASTR", "EP", "MED"],
        savePath=save_model_path,
        saveName="best_model_ROC_validation",
        draw=False,
    )
    utilities.plotPR(
        GT=Yval_categorical,
        PRED=Pval_softmax,
        classes=["Not_tumor", "Tumor"]
        if args_dict["NBR_CLASSES"] == 2
        else ["ASTR", "EP", "MED"],
        savePath=save_model_path,
        saveName="best_model_PR_validation",
        draw=False,
    )

    # ## SAVE FINAL CURVES
    print(f'{" "*6}Saving training curves and tabular evaluation data...')
    fig, ax = plt.subplots(
        figsize=(20, 15),
        nrows=3 if history.history["MatthewsCorrelationCoefficient"] else 2,
        ncols=1,
    )
    # print training loss
    ax[0].plot(history.history["loss"], label="training loss")
    ax[0].plot(history.history["val_loss"], label="validation loss")
    ax[0].set_title(f"Test loss")
    ax[0].legend()
    # print training accuracy
    ax[1].plot(history.history["accuracy"], label="training accuracy")
    ax[1].plot(history.history["val_accuracy"], label="validation accuracy")
    ax[1].set_title(
        f'Test MCC -> (last)  {summary_test[str(cv_f+1)]["last"]["overall_accuracy"]:0.3f}, (best) {summary_test[str(cv_f+1)]["best"]["overall_accuracy"]:0.3f}'
    )
    ax[1].legend()

    # print training MCC
    if history.history["MatthewsCorrelationCoefficient"]:
        ax[2].plot(
            history.history["MatthewsCorrelationCoefficient"], label="training MCC"
        )
        ax[2].plot(
            history.history["val_MatthewsCorrelationCoefficient"],
            label="validation MCC",
        )
        ax[2].set_title(
            f'Test accuracy -> (last)  {summary_test[str(cv_f+1)]["last"]["matthews_correlation_coefficient"]:0.3f}, (best) {summary_test[str(cv_f+1)]["best"]["matthews_correlation_coefficient"]:0.3f}'
        )
        ax[2].legend()

    fig.savefig(os.path.join(save_model_path, "training_curves.png"))
    plt.close(fig)

    ## SAVE MODEL PORFORMANCE FOR for THIS fold
    for m in ["last", "best"]:
        filename = os.path.join(
            args_dict["SAVE_PATH"],
            f"fold_{str(cv_f+1)}",
            f"{m}_summary_evaluation.txt",
        )
        accs = summary_test[str(cv_f + 1)][m]["overall_accuracy"] * 100
        np.savetxt(filename, [accs], fmt="%.4f")

    # SAVE PER METRICS AS CSV
    summary_file = os.path.join(
        args_dict["SAVE_PATH"],
        f"fold_{str(cv_f+1)}",
        f"tabular_test_summary.csv",
    )
    csv_file = open(summary_file, "w")
    writer = csv.writer(csv_file)
    csv_header = [
        "classification_type",
        "nbr_classes",
        "model_type",
        "model_version",
        "fold",
        "precision",
        "recall",
        "accuracy",
        "f1-score",
        "auc",
        "matthews_correlation_coefficient",
    ]
    writer.writerow(csv_header)
    # build rows to save in the csv file
    csv_rows = []
    for m in ["last", "best"]:
        csv_rows.append(
            [
                "tumor-vs-no_tumor",
                2,
                "2D_detection_model",
                m,
                cv_f + 1,
                summary_test[str(cv_f + 1)][m]["overall_precision"],
                summary_test[str(cv_f + 1)][m]["overall_recall"],
                summary_test[str(cv_f + 1)][m]["overall_accuracy"],
                summary_test[str(cv_f + 1)][m]["overall_f1-score"],
                summary_test[str(cv_f + 1)][m]["overall_auc"],
                summary_test[str(cv_f + 1)][m]["matthews_correlation_coefficient"],
            ]
        )
    writer.writerows(csv_rows)
    csv_file.close()
    print(f'{" "*6}Done! To the next fold.')
## SAVE SUMMARY FOR ALL THE FOLDS IN ONE FILE
summary_file = os.path.join(args_dict["SAVE_PATH"], f"tabular_test_summary.csv")
csv_file = open(summary_file, "w")
writer = csv.writer(csv_file)
csv_header = [
    "classification_type",
    "nbr_classes",
    "model_type",
    "model_version",
    "fold",
    "precision",
    "recall",
    "accuracy",
    "f1-score",
    "auc",
    "matthews_correlation_coefficient",
]
writer.writerow(csv_header)
# build rows to save in the csv file
csv_rows = []
for cv_f in range(args_dict["NBR_FOLDS"]):
    for m in ["last", "best"]:
        csv_rows.append(
            [
                "tumor-vs-no_tumor",
                2,
                m,
                "DetectionModel",
                cv_f + 1,
                summary_test[str(cv_f + 1)][m]["overall_precision"],
                summary_test[str(cv_f + 1)][m]["overall_recall"],
                summary_test[str(cv_f + 1)][m]["overall_accuracy"],
                summary_test[str(cv_f + 1)][m]["overall_f1-score"],
                summary_test[str(cv_f + 1)][m]["overall_auc"],
                summary_test[str(cv_f + 1)][m]["matthews_correlation_coefficient"],
            ]
        )
writer.writerows(csv_rows)
csv_file.close()

# %%
