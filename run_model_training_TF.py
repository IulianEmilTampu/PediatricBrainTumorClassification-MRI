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

from sklearn.model_selection import train_test_split, StratifiedKFold
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
        "-da",
        "--DATA_AUGMENTATION",
        required=False,
        default=True,
        type=bool,
        help="Specify if data augmentation should be applied",
    )
    parser.add_argument(
        "-dn",
        "--DATA_NORMALIZATION",
        required=False,
        default=True,
        type=bool,
        help="Specify if data normalization should be performed (check pretrained models if they need it)",
    )
    parser.add_argument(
        "-ds",
        "--DATA_SCALE",
        required=False,
        default=False,
        type=bool,
        help="Specify if data should be scaled (from [0,255] to [0,1]) (check pretrained models if they need it)",
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
        "-freeze_weights",
        "--FREEZE_WEIGHTS",
        required=False,
        dest="FREEZE_WEIGHTS",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Specify if pretrained model weights should be frozen.",
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
        "-age_encoder_version",
        "--AGE_ENCODER_VERSION",
        required=False,
        type=str,
        default="no_encoder",
        help="Available age encoders: no_encoder | simple_age_encode | large_age_encoder",
    )
    parser.add_argument(
        "-age_normalization",
        "--AGE_NORMALIZATION",
        required=False,
        dest="AGE_NORMALIZATION",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Specify if the age should be normalized. If True, age is normalized using mean and std from the trianing dataset ([-1,1] norm).",
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
    if os.name == "posix":
        args_dict = {
            "WORKING_FOLDER": "/flush/iulta54/Research/P5-MICCAI2023",
            "IMG_DATASET_FOLDER": "/flush/iulta54/Research/Data/CBTN/EXTRACTED_SLICES_TFR_MERGED_FROM_TB_20230320",
            # "IMG_DATASET_FOLDER": "/flush/iulta54/Research/Data/BRATS/extracted_slices/2021/saved_images",
            "DATASET_TYPE": "CBTN",
            "NBR_CLASSES": 3,
            "GPU_NBR": "0",
            "NBR_FOLDS": 1,
            "LEARNING_RATE": 0.0001,
            "BATCH_SIZE": 32,
            "MAX_EPOCHS": 5,
            "DATA_AUGMENTATION": True,
            "DATA_NORMALIZATION": False,
            "DATA_SCALE": False,
            "USE_PRETRAINED_MODEL": False,
            "PATH_TO_PRETRAINED_MODEL": "/flush/iulta54/Research/P5-MICCAI2023/trained_models_archive/TEST_pretraining_optm_ADAM_SDM4_TFRdata_False_modality_T2_loss_MCC_and_CCE_Loss_lr_0.0001_batchSize_32_pretrained_False_useAge_False_useGradCAM_False_seed_20091229/fold_1/last_model/last_model",
            "FREEZE_WEIGHTS": False,
            "USE_AGE": False,
            "AGE_NORMALIZATION": False,
            "AGE_ENCODER_VERSION": "simple_age_encoder",
            "USE_GRADCAM": False,
            "LOSS": "MCC_and_CCE_Loss",
            "RANDOM_SEED_NUMBER": 1111,
            "MR_MODALITIES": ["T2"],
            "DEBUG_DATASET_FRACTION": 1,
            "TFR_DATA": True,
            "MODEL_TYPE": "SDM4",
            "MODEL_NAME": "TEST_check_run_time",
            "OPTIMIZER": "ADAM",
        }
    else:
        args_dict = {
            "WORKING_FOLDER": r"C:\Users\iulta54\Documents\PediatricBrainTumorClassification",
            "IMG_DATASET_FOLDER": r"C:\Datasets\CBTN\EXTRACTED_SLICES_TFR_MERGED_FROM_TB_20230320",
            # "IMG_DATASET_FOLDER": "/flush/iulta54/Research/Data/BRATS/extracted_slices/2021/saved_images",
            "DATASET_TYPE": "CBTN",
            "NBR_CLASSES": 3,
            "GPU_NBR": "0",
            "NBR_FOLDS": 1,
            "LEARNING_RATE": 0.0001,
            "BATCH_SIZE": 32,
            "MAX_EPOCHS": 50,
            "DATA_AUGMENTATION": True,
            "DATA_NORMALIZATION": True,
            "DATA_SCALE": True,
            "USE_PRETRAINED_MODEL": False,
            "PATH_TO_PRETRAINED_MODEL": "/flush/iulta54/Research/P5-MICCAI2023/trained_models_archive/TEST_pretraining_optm_ADAM_SDM4_TFRdata_False_modality_T2_loss_MCC_and_CCE_Loss_lr_0.0001_batchSize_32_pretrained_False_useAge_False_useGradCAM_False_seed_20091229/fold_1/last_model/last_model",
            "FREEZE_WEIGHTS": True,
            "USE_AGE": True,
            "AGE_NORMALIZATION": True,
            "AGE_ENCODER_VERSION": "no_encoder",
            "USE_GRADCAM": False,
            "LOSS": "MCC_and_CCE_Loss",
            "RANDOM_SEED_NUMBER": 1111,
            "MR_MODALITIES": ["T2"],
            "DEBUG_DATASET_FRACTION": 1,
            "TFR_DATA": True,
            "MODEL_TYPE": "SDM4",
            "MODEL_NAME": "TEST_OVERSAMPLING_EP",
            "OPTIMIZER": "ADAM",
        }

# revise model name
args_dict[
    "MODEL_NAME"
] = f'{args_dict["MODEL_NAME"]}_optm_{args_dict["OPTIMIZER"]}_{args_dict["MODEL_TYPE"]}_TFRdata_{args_dict["TFR_DATA"]}_modality_{"_".join([i for i in args_dict["MR_MODALITIES"]])}_loss_{args_dict["LOSS"]}_lr_{args_dict["LEARNING_RATE"]}_batchSize_{args_dict["BATCH_SIZE"]}_pretrained_{args_dict["USE_PRETRAINED_MODEL"]}_frozenWeight_{args_dict["FREEZE_WEIGHTS"]}_useAge_{args_dict["USE_AGE"]}_{args_dict["AGE_ENCODER_VERSION"]}_useGradCAM_{args_dict["USE_GRADCAM"]}_seed_{args_dict["RANDOM_SEED_NUMBER"]}'

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
    dataset_type=args_dict["DATASET_TYPE"],
    modalities=args_dict["MR_MODALITIES"],
    return_labels=True,
    task="detection" if args_dict["NBR_CLASSES"] == 2 else "classification",
    nbr_classes=args_dict["NBR_CLASSES"],
    tumor_min_rpl=10,
    tumor_max_rpl=90,
    brain_min_rpl=1,
    brain_max_rpl=100,
    file_format="tfrecords" if args_dict["TFR_DATA"] else "jpeg",
    tumor_loc=["infra", "supra"],
)

unique_patien_IDs_with_labels = list(
    dict.fromkeys([(f[1], f[2]) for f in all_file_names])
)
unique_patien_IDs_labels = [f[1] for f in unique_patien_IDs_with_labels]

args_dict["NBR_SUBJECTS"] = len(unique_patien_IDs_with_labels)

subj_train_val_idx, subj_test_idx = train_test_split(
    unique_patien_IDs_with_labels,
    stratify=unique_patien_IDs_labels
    if not all([args_dict["NBR_CLASSES"] == 2, args_dict["DATASET_TYPE"] == "BRATS"])
    else None,
    test_size=0.20,
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
if not all([args_dict["NBR_CLASSES"] == 2, args_dict["DATASET_TYPE"] == "BRATS"]):
    for c in range(args_dict["NBR_CLASSES"]):
        args_dict["CLASS_WEIGHTS"][c] = class_weights_values[c] ** 2
else:
    for c in range(args_dict["NBR_CLASSES"]):
        args_dict["CLASS_WEIGHTS"][c] = 1


# # OVERSAMPLING THE EP class
# random.seed(args_dict["RANDOM_SEED_NUMBER"])
# EP_samples = [i for i in subj_train_val_idx if i[1] == 1]
# subj_train_val_idx.extend(random.choices(EP_samples, k=50))
# subj_train_val_idx_labels = [f[1] for f in subj_train_val_idx]

subj_train_idx, subj_val_idx = [], []
per_fold_training_files, per_fold_validation_files = [], []
# set cross validation
if args_dict["NBR_FOLDS"] > 1:
    kf = StratifiedKFold(
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
        stratify=subj_train_val_idx_labels
        if not all(
            [args_dict["NBR_CLASSES"] == 2, args_dict["DATASET_TYPE"] == "BRATS"]
        )
        else None,
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
check_test_files = False
if check_test_files:
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
else:
    print(f"WARNING!!!\nSKipping check of test file.")

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
        batch_size=args_dict["BATCH_SIZE"],
        buffer_size=1,
        return_gradCAM=False,
        return_age=False,
        dataset_type="test",
        nbr_classes=args_dict["NBR_CLASSES"],
        output_as_RGB=False,
    )

    img_stats, gradCAM_stats, age_stats = data_utilities.get_normalization_values(
        gen, gen_steps, return_age_norm_values=False, return_gradCAM_norm_values=False
    )

    print(f"Image data (mean+-std): {img_stats}")
    print(f"gradCAM data (mean+-std): {gradCAM_stats}")
    print(f"Age data (mean+-std): {age_stats}")

    if Path(test_files[0]).suffix == ".tfrecords":
        show_batched_example_tfrs_dataset(
            gen,
            nbr_images=5,
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
    elif any([args_dict["USE_AGE"], args_dict["USE_GRADCAM"]]):
        raise ValueError(
            "Trying to run training using dataset from .jpeg files while asking for age information and/or gradCAM.\nThis is not yet implemented. Use TFR dataset."
        )
    else:
        data_gen = data_utilities.img_data_generator

    # define generator parameters
    target_size = (224, 224)

    # # ################## TRAINING GENERATOR (get also normalization values)
    tr_files = per_fold_training_files[cv_f]
    random.Random(args_dict["RANDOM_SEED_NUMBER"]).shuffle(tr_files)

    # get norm stat if needed
    if args_dict["DATA_NORMALIZATION"]:
        train_gen, train_steps = data_gen(
            file_paths=tr_files[
                0 : int(len(tr_files) * args_dict["DEBUG_DATASET_FRACTION"])
            ],
            input_size=target_size,
            batch_size=args_dict["BATCH_SIZE"],
            buffer_size=1000,
            return_gradCAM=args_dict["USE_GRADCAM"],
            return_age=args_dict["USE_AGE"],
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
        norm_stats = [None, None, None]

    # build actuall training datagen with normalized values
    output_as_RGB = (
        True
        if any(
            [
                args_dict["MODEL_TYPE"] == "EfficientNet",
                args_dict["MODEL_TYPE"] == "ResNet50",
                args_dict["MODEL_TYPE"] == "VGG16",
            ]
        )
        else False
    )

    train_gen, train_steps = data_gen(
        file_paths=tr_files[
            0 : int(len(tr_files) * args_dict["DEBUG_DATASET_FRACTION"])
        ],
        input_size=target_size,
        batch_size=args_dict["BATCH_SIZE"],
        buffer_size=500,
        return_gradCAM=args_dict["USE_GRADCAM"],
        return_age=args_dict["USE_AGE"],
        dataset_type="train",
        nbr_classes=args_dict["NBR_CLASSES"],
        output_as_RGB=output_as_RGB,
    )
    # plot also histogram of values
    if args_dict["TFR_DATA"]:
        data_utilities.plot_tfr_dataset_intensity_dist(
            train_gen, train_steps, plot_name="Training_data", save_path=save_model_path
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
        return_gradCAM=args_dict["USE_GRADCAM"],
        return_age=args_dict["USE_AGE"],
        dataset_type="val",
        nbr_classes=args_dict["NBR_CLASSES"],
        output_as_RGB=output_as_RGB,
    )
    # plot also histogram of values
    if args_dict["TFR_DATA"]:
        data_utilities.plot_tfr_dataset_intensity_dist(
            val_gen, val_steps, plot_name="Validation_data", save_path=save_model_path
        )
    print(f'{" "*6}Validation gen. done!')

    test_gen, test_steps = data_gen(
        file_paths=test_files,
        input_size=target_size,
        batch_size=args_dict["BATCH_SIZE"],
        buffer_size=10,
        return_gradCAM=args_dict["USE_GRADCAM"],
        return_age=args_dict["USE_AGE"],
        dataset_type="test",
        nbr_classes=args_dict["NBR_CLASSES"],
        output_as_RGB=output_as_RGB,
    )
    # plot also histogram of values
    if args_dict["TFR_DATA"]:
        data_utilities.plot_tfr_dataset_intensity_dist(
            test_gen, test_steps, plot_name="Test_data", save_path=save_model_path
        )
    print(f'{" "*6}Testing gen. done!')

    # -------------
    # CREATE MODEL
    # ------------
    importlib.reload(models)
    print(f'{" "*3}Building model...')
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
            image_normalization_stats=norm_stats[0],
            scale_image=args_dict["DATA_SCALE"],
            data_augmentation=args_dict["DATA_AUGMENTATION"],
            kernel_size=(3, 3),
            pool_size=(2, 2),
            use_age=args_dict["USE_AGE"],
            age_normalization_stats=norm_stats[2]
            if args_dict["AGE_NORMALIZATION"]
            else None,
            age_encoder_version=args_dict["AGE_ENCODER_VERSION"],
            use_pretrained=args_dict["USE_PRETRAINED_MODEL"],
            pretrained_model_path=args_dict["PATH_TO_PRETRAINED_MODEL"],
            freeze_weights=args_dict["FREEZE_WEIGHTS"],
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
            input_shape=(input_shape[0], input_shape[1], 3),
            use_age=args_dict["USE_AGE"],
            use_age_thr_tabular_network=False,
            pretrained=args_dict["USE_PRETRAINED_MODEL"],
            freeze_weights=args_dict["FREEZE_WEIGHTS"],
        )
    elif args_dict["MODEL_TYPE"] == "VGG16":
        print(f'{" "*6}Using {args_dict["MODEL_TYPE"]} model.')
        model = models.VGG16(
            num_classes=args_dict["NBR_CLASSES"],
            input_shape=(input_shape[0], input_shape[1], 3),
            use_age=args_dict["USE_AGE"],
            use_age_thr_tabular_network=False,
            pretrained=args_dict["USE_PRETRAINED_MODEL"],
            freeze_weights=args_dict["FREEZE_WEIGHTS"],
        )
    elif args_dict["MODEL_TYPE"] == "ResNet50":
        print(f'{" "*6}Using {args_dict["MODEL_TYPE"]} model.')
        model = models.ResNet50(
            num_classes=args_dict["NBR_CLASSES"],
            input_shape=(input_shape[0], input_shape[1], 3),
            use_age=args_dict["USE_AGE"],
            use_age_thr_tabular_network=False,
            pretrained=args_dict["USE_PRETRAINED_MODEL"],
            freeze_weights=args_dict["FREEZE_WEIGHTS"],
        )
    else:
        raise ValueError(
            "Model type not among the ones that are implemented.\nDefine model in the models.py file and add code here for building the model."
        )

    # ################################# COMPILE MODEL

    # learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    #     args_dict["LEARNING_RATE"], args_dict["MAX_EPOCHS"], 0, power=0.99
    # )

    # values obtained using the LRFind callback

    learning_rate_fn = tfa.optimizers.CyclicalLearningRate(
        initial_learning_rate=1e-2,
        maximal_learning_rate=1e-1,
        scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
        step_size=2 * train_steps,
    )

    # learning_rate_fn = None

    # plot learning rate
    if isinstance(
        learning_rate_fn, tfa.optimizers.cyclical_learning_rate.CyclicalLearningRate
    ):
        fig = plt.figure()
        step = np.arange(0, args_dict["MAX_EPOCHS"] * train_steps)
        lr = learning_rate_fn(step)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        ax1.plot(step, lr)
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Learning Rate")
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks([i * train_steps for i in range(args_dict["MAX_EPOCHS"])])
        ax2.set_xticklabels([i for i in range(args_dict["MAX_EPOCHS"])])
        ax2.set_xlabel("Epochs")
        # save figure
        fig.savefig(
            os.path.join(os.path.dirname(save_model_path), "learning_rate_curve.png")
        )
        plt.close(fig)

    if args_dict["OPTIMIZER"] == "SGD":
        print(f'{" "*6}Using SGD optimizer.')
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate_fn
            if learning_rate_fn
            else args_dict["LEARNING_RATE"],
            decay=1e-6,
            momentum=0.9,
            nesterov=True,
            clipvalue=0.5,
        )

    elif args_dict["OPTIMIZER"] == "ADAM":
        print(f'{" "*6}Using AdamW optimizer.')

        optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate_fn
            if learning_rate_fn
            else args_dict["LEARNING_RATE"],
            weight_decay=0.0001,
        )

    # wrap using LookAhead which helps smoothing out validation curves
    optimizer = Lookahead(optimizer, sync_period=5, slow_step_size=0.5)

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
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=os.path.join(save_model_path, "model_architecture.jpeg"),
            show_shapes=True,
            show_layer_activations=True,
            expand_nested=True,
        )
    except:
        print("Failed to plot model architecture.")

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
        tf_callbacks.SaveBestModelWeights(
            save_path=best_model_path, monitor="val_loss", mode="min"
        ),
        # model_checkpoint_callback,
        tf_callbacks.LossAndErrorPrintingCallback(
            save_path=save_model_path, print_every_n_epoch=5
        ),
    ]

    # save training configuration (right before training to account for the changes made in the meantime)
    args_dict["OPTIMIZER"] = str(type(optimizer))
    args_dict["LOSS_TYPE"] = str(type(loss))
    args_dict["LEARNING_SCHEDULER"] = str(
        (type(learning_rate_fn) if learning_rate_fn else "constant")
    )

    with open(os.path.join(args_dict["SAVE_PATH"], "config.json"), "w") as config_file:
        config_file.write(json.dumps(args_dict))

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
    ax[0].set_title(f"Train and validation loss")
    ax[0].legend()
    # print training accuracy
    ax[1].plot(history.history["accuracy"], label="training accuracy")
    ax[1].plot(history.history["val_accuracy"], label="validation accuracy")
    ax[1].set_title("Train and Validation accuracy")
    # ax[1].set_title(
    #     f'Test accuracy -> (last)  {summary_test[str(cv_f+1)]["last"]["overall_accuracy"]:0.3f}, (best) {summary_test[str(cv_f+1)]["best"]["overall_accuracy"]:0.3f}'
    # )
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
        ax[2].set_title("Train and Validation MCC")
        # ax[2].set_title(
        #     f'Test MCC -> (last)  {summary_test[str(cv_f+1)]["last"]["matthews_correlation_coefficient"]:0.3f}, (best) {summary_test[str(cv_f+1)]["best"]["matthews_correlation_coefficient"]:0.3f}'
        # )
        ax[2].legend()

    fig.savefig(os.path.join(save_model_path, "training_curves.png"))
    plt.close(fig)

# ------------------
# MODEL EVALUATION
# ------------------
import test

importlib.reload(test)

args_dict["CV_MODEL_FOLDER"] = args_dict["SAVE_PATH"]
test.run_testing(args_dict)

# %%
