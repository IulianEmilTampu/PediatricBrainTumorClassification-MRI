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
import glob
import csv
import json
import numpy as np
import argparse
import importlib
import logging
import random
from pathlib import Path

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils.extmath import softmax

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms, utils
# from torch.utils.data import Dataset, DataLoader

# local imports
import utilities
import data_utilities
import models

# import trainer

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
        default="myModel",
        help="Name used to save the model and the scores",
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
        "IMG_DATASET_FOLDER": "/flush/iulta54/Research/random_stuf/Extract_pngs_from_brats/transversal_BraTS2020",
        "GPU_NBR": "0",
        "MODEL_NAME": "SDM4_t1_t2_BraTS_fullDataset_lr10em6_more_data",
        "NBR_FOLDS": 5,
        "LEARNING_RATE": 0.000001,
        "BATCH_SIZE": 16,
        "MAX_EPOCHS": 50,
        "RANDOM_SEED_NUMBER": 29122009,
    }

# --------------------------------------
# set GPU (or device)
# --------------------------------------

# import tensorflow
import tensorflow as tf
import warnings

tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.utils import to_categorical
from tensorflow_addons.optimizers import Lookahead

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

devices = tf.config.list_physical_devices("GPU")

if devices:
    print(f'Running training on GPU # {args_dict["GPU_NBR"]} \n')
    warnings.simplefilter(action="ignore", category=FutureWarning)
    tf.config.experimental.set_memory_growth(devices[0], True)
else:
    Warning(
        f"ATTENTION!!! MODEL RUNNING ON CPU. Check implementation in case GPU is wanted."
    )


# # check if cuda is available
# if torch.cuda.is_available():
#     # set device to the one specified
#     print(f'Running training on GPU # {args_dict["GPU_NBR"]} \n')
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = args_dict["GPU_NBR"]
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# elif torch.backends.mps.is_available():
#     print(f'Running training on GPU # {args_dict["GPU_NBR"]} \n')
#     DEVICE = "mps"
# else:
#     DEVICE = "cpu"
#     Warning(
#         f"ATTENTION!!! MODEL RUNNING ON CPU. Check implementation in case GPU is wanted."
#     )
# args_dict["DEVICE"] = DEVICE

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


if not su_debug_flag:
    # save training configuration
    with open(os.path.join(args_dict["SAVE_PATH"], "config.json"), "w") as config_file:
        config_file.write(json.dumps(args_dict))

# print input variables
max_len = max([len(key) for key in args_dict])
[print(f"{key:{max_len}s}: {value}") for key, value in args_dict.items()]

print(f"Splitting dataset (per-volume (subject) splitting).")
"""
Independently from the combination of modalities, the test validation and train sets
are defined so that no vlomume is present in more than one set.

Steps
2 - using the number of subject, screate indexes to identify which files
    are used for training, validation and testing
3 - save the information about the split.
"""

# ######### this it the default on
# # get the unique subjects in the IMG_DATASET_FOLDER
# # NOTE that the heuristic used to get the unique patient IDs might be changed depending on the dataset
# all_file_names = glob.glob(os.path.join(args_dict["IMG_DATASET_FOLDER"], "*.jpeg"))


# ################ THIS IS SPECIAL JUST TO MAKE IT WORK
# get images from all the modalities for training
all_file_names = []
for modality in ["t1", "t2"]:
    for s in ["slices_without_tumor_label_0", "slices_with_tumor_label_1"]:
        files = glob.glob(
            os.path.join(args_dict["IMG_DATASET_FOLDER"], modality, s, "*.jpeg")
        )
        if s == "slices_with_tumor_label_1":
            # filter files with relative position of the tumor within [20, 80]
            min_rlp = 20
            max_rlp = 80
            files = [
                (f, int(os.path.basename(f).split("_")[2]))
                for f in files
                if all(
                    [
                        float(os.path.basename(f).split("_")[5]) >= min_rlp,
                        float(os.path.basename(f).split("_")[5]) <= max_rlp,
                    ]
                )
            ]
        if s == "slices_without_tumor_label_0":
            # filter files with relative position of the tumor within [20, 80]
            min_rlp = 1
            max_rlp = 25
            files = [
                (f, int(os.path.basename(f).split("_")[2]))
                for f in files
                if all(
                    [
                        float(os.path.basename(f).split("_")[5]) >= min_rlp,
                        float(os.path.basename(f).split("_")[5]) <= max_rlp,
                    ]
                )
            ]
        all_file_names.extend(files)

random.shuffle(all_file_names)
unique_patien_IDs = list(dict.fromkeys(f[1] for f in all_file_names))

######### DEBUG
# random.shuffle(unique_patien_IDs)
# unique_patien_IDs = unique_patien_IDs[0:25]
######### end

args_dict["NBR_SUBJECTS"] = len(unique_patien_IDs)

subj_train_val_idx, subj_test_idx = train_test_split(
    unique_patien_IDs, test_size=0.05, random_state=args_dict["RANDOM_SEED_NUMBER"]
)
test_files = [f[0] for f in all_file_names if any([i == f[1] for i in subj_test_idx])]
print(f'{"# Train-val subjects":18s}: {len(subj_train_val_idx):2d}')
print(
    f'{"# Test subjects":18s}: {len(subj_test_idx):2d} ({subj_test_idx} {len(test_files)} total images)'
)

subj_train_idx, subj_val_idx = [], []
per_fold_training_files, per_fold_validation_files = [], []
# set cross validation
if args_dict["NBR_FOLDS"] > 1:
    kf = KFold(
        n_splits=args_dict["NBR_FOLDS"],
        shuffle=True,
        random_state=args_dict["RANDOM_SEED_NUMBER"],
    )
    for idx, (train_index, val_index) in enumerate(kf.split(subj_train_val_idx)):
        subj_train_idx.append([subj_train_val_idx[i] for i in train_index])
        subj_val_idx.append([subj_train_val_idx[i] for i in val_index])
        # get also the respective training and validation file names for the generator
        per_fold_training_files.append(
            [
                f[0]
                for f in all_file_names
                if any([i == f[1] for i in subj_train_idx[-1]])
            ]
        )
        per_fold_validation_files.append(
            [f[0] for f in all_file_names if any([i == f[1] for i in subj_val_idx[-1]])]
        )

        # print to check that all is good
        print(
            f'Fold {idx+1}: \n {""*4}{"training":10s} ->{subj_train_idx[-1]} \n {""*4}{"validation":10s} ->{subj_val_idx[-1]}'
        )
else:
    # N_FOLDS is only one, setting 10% of the training dataset as validation
    print("DEBUG: getting indexes of training and validation files for one fold.")
    aus_train, aus_val = train_test_split(
        subj_train_val_idx, test_size=0.1, random_state=args_dict["RANDOM_SEED_NUMBER"]
    )
    subj_train_idx.append(aus_train)
    subj_val_idx.append(aus_val)

    # print(f"DEBUG: training indexes {subj_train_idx}.")
    # print(f"DEBUG: val indexes {subj_val_idx}.")

    per_fold_training_files.append(
        [f[0] for f in all_file_names if any([i == f[1] for i in subj_train_idx[-1]])]
    )
    per_fold_validation_files.append(
        [f[0] for f in all_file_names if any([i == f[1] for i in subj_val_idx[-1]])]
    )

    # print to check that all is good
    print(
        f'Fold {args_dict["NBR_FOLDS"]}: \n {""*4}{"training":10s} ->{subj_train_idx[-1]} ({len(per_fold_training_files[-1])} images) \n {""*4}{"validation":10s} ->{subj_val_idx[-1]} ({len(per_fold_validation_files[-1])} images)'
    )

# check that no testing files are in the training or validation
for idx, test_f in enumerate(test_files):
    print(
        f"Checking test files ({idx+1:0{len(str(len(test_files)))}d}\{len(test_files)})\r",
        end="",
    )
    # check in each fold
    for fold in range(len(per_fold_training_files)):
        # check in the training set
        if any([test_f == f for f in per_fold_training_files[fold]]):
            raise ValueError(
                f"ATTENTION!!!\nSome of the testing files are part of the training set!\nCheck implementation"
            )
        if any([test_f == f for f in per_fold_validation_files[fold]]):
            raise ValueError(
                f"ATTENTION!!!\nSome of the testing files are part of the training set!\nCheck implementation"
            )

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
# %%
# ---------
# RUNIING CROSS VALIDATION TRAINING
# ---------

# create dictionary where to save the test performance
summary_test = {}

for cv_f in range(args_dict["NBR_FOLDS"]):
    # make forder where to save the model
    save_model_path = os.path.join(args_dict["SAVE_PATH"], f"fold_{cv_f+1}")
    Path(save_model_path).mkdir(parents=True, exist_ok=True)
    summary_test[str(cv_f + 1)] = {"best": [], "last": []}

    print(f'{" "*3}Setting up training an validation data Generators ...')

    # --------------------------
    # CREATE DATA GENERATORS
    # -------------------------
    importlib.reload(data_utilities)

    target_size = (224, 224)
    train_gen = data_utilities.get_data_generator_TF(
        sample_files=per_fold_training_files[cv_f],
        target_size=target_size,
        batch_size=args_dict["BATCH_SIZE"],
        dataset_type="training",
    )
    val_gen = data_utilities.get_data_generator_TF(
        sample_files=per_fold_validation_files[cv_f],
        target_size=target_size,
        batch_size=args_dict["BATCH_SIZE"],
        dataset_type="validation",
    )
    test_gen = data_utilities.get_data_generator_TF(
        sample_files=test_files,
        target_size=target_size,
        batch_size=args_dict["BATCH_SIZE"],
        dataset_type="testing",
    )

    print(
        f"Training: {len(train_gen)}\nValidation: {len(val_gen)}\nTesting: {len(test_gen)}"
    )

    ## BUILD DETERCTION MODEL
    importlib.reload(models)
    print(f'{" "*3}Building model ...')
    # build custom model (WHAT HAS BEEN USED IN THE qMRI PROJECT)
    model = models.SimpleDetectionModel_TF(
        num_classes=2,
        input_shape=(224, 224, 1),
        class_weights=None,
        kernel_size=(3, 3),
        pool_size=(2, 2),
        model_name="SimpleDetectionModel",
    )

    ## COMPILE MODEL
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        args_dict["LEARNING_RATE"], args_dict["MAX_EPOCHS"], 0, power=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args_dict["LEARNING_RATE"])
    optimizer = Lookahead(optimizer, sync_period=5, slow_step_size=0.5)

    loss = tf.keras.losses.BinaryCrossentropy()

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    ## SET MODEL CHECKPOINT
    best_model_path = os.path.join(save_model_path, "best_model_weights", "")
    Path(best_model_path).mkdir(parents=True, exist_ok=True)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )

    ## RUN MODEL TRAINING
    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        shuffle=True,
        validation_data=val_gen,
        validation_steps=len(val_gen),
        epochs=args_dict["MAX_EPOCHS"],
        verbose=1,
        callbacks=[model_checkpoint_callback],
    )

    args_dict["MAX_EPOCHS"]
    # save last model
    model.save(os.path.join(save_model_path, "last_model"))

    ## EVALUATE LAST & BEST MODEL
    importlib.reload(utilities)
    # ###################### LAST MODEL
    # get the per_slice classification
    Ptest_softmax = []
    Ytest_categorical = []
    for i in range(len(test_gen)):
        x, y = next(iter(test_gen))

        Ytest_categorical.append(y)

        Ptest_softmax.append(model.predict(x))

    Ptest_softmax = np.row_stack(Ptest_softmax)
    Ptest = np.argmax(Ptest_softmax, axis=-1)

    Ytest_categorical = np.row_stack(Ytest_categorical)
    summary_test[str(cv_f + 1)]["last"] = utilities.get_performance_metrics(
        Ytest_categorical, Ptest_softmax, average="macro"
    )
    # [print(f'{key}: {value}\n') for key, value in summary_test['last'].items()]
    summary_test[str(cv_f + 1)]["last"]["per_case_prediction"] = Ptest

    # ###################### BEST MODEL
    model.load_weights(best_model_path)
    # get the per_slice classification
    Ptest_softmax = []
    Ytest_categorical = []
    for i in range(len(test_gen)):
        x, y = next(iter(test_gen))

        Ytest_categorical.append(y)

        Ptest_softmax.append(model.predict(x))

    Ptest_softmax = np.row_stack(Ptest_softmax)
    Ptest = np.argmax(Ptest_softmax, axis=-1)

    Ytest_categorical = np.row_stack(Ytest_categorical)
    summary_test[str(cv_f + 1)]["best"] = utilities.get_performance_metrics(
        Ytest_categorical, Ptest_softmax, average="macro"
    )
    # [print(f'{key}: {value}\n') for key, value in summary_test['last'].items()]
    summary_test[str(cv_f + 1)]["best"]["per_case_prediction"] = Ptest

    ## SAVE TRAINING CURVES

    fig, ax = plt.subplots(figsize=(20, 15), nrows=2, ncols=1)
    # print training loss
    ax[0].plot(history.history["loss"], label="training loss")
    ax[0].plot(history.history["val_loss"], label="validation loss")
    ax[0].set_title(f"Test loss")
    ax[0].legend()
    # print training accuracy
    ax[1].plot(history.history["accuracy"], label="training accuracy")
    ax[1].plot(history.history["val_accuracy"], label="validation accuracy")
    ax[1].set_title(
        f'Test accuracy -> (last)  {summary_test[str(cv_f+1)]["last"]["overall_accuracy"]:0.3f}, (best) {summary_test[str(cv_f+1)]["best"]["overall_accuracy"]:0.3f}'
    )
    ax[1].legend()
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
