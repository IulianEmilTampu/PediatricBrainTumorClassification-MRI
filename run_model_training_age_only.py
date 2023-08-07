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
import sys

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
        "-dn",
        "--DATA_NORMALIZATION",
        required=False,
        default=True,
        type=bool,
        help="Specify if data normalization should be performed (check pretrained models if they need it)",
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
    parser.add_argument(
        "-model_version",
        "--MODEL_VERSION",
        required=False,
        type=str,
        default="age_to_classes",
        help="Available model versions: age_to_classes | simple_age_encode | large_age_encoder",
    )

    args_dict = dict(vars(parser.parse_args()))

else:
    # # # # # # # # # # # # # # DEBUG
    args_dict = {
        "WORKING_FOLDER": "/flush/iulta54/Research/P5-Pediatric_tumor_classification",
        "IMG_DATASET_FOLDER": "/flush/iulta54/Research/Data/CBTN_v1/EXTRACTED_SLICES_TFR_MERGED_FROM_TB_20230320",
        "DATASET_TYPE": "CBTN",
        "MR_MODALITIES": ["T2"],
        "NBR_CLASSES": 3,
        "GPU_NBR": "0",
        "NBR_FOLDS": 1,
        "OPTIMIZER": "ADAM",
        "LEARNING_RATE": 0.001,
        "LOSS": "CCE",
        "BATCH_SIZE": 8,
        "MAX_EPOCHS": 5,
        "MODEL_VERSION": "2D-SDM4",
        "DATA_NORMALIZATION": True,
        "RANDOM_SEED_NUMBER": 20091229,
        "DEBUG_DATASET_FRACTION": 1,
        "MODEL_NAME": "TEST",
    }

# revise model name
args_dict[
    "MODEL_NAME"
] = f'{args_dict["MODEL_NAME"]}_{args_dict["MODEL_VERSION"]}_optm_normalization_{args_dict["DATA_NORMALIZATION"]}_loss_{args_dict["LOSS"]}_lr_{args_dict["LEARNING_RATE"]}_batchSize_{args_dict["BATCH_SIZE"]}_seed_{args_dict["RANDOM_SEED_NUMBER"]}'

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
2 - using the number of subject, create indexes to identify which files
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

# %%
# ---------
# RUNIING CROSS VALIDATION TRAINING
# ---------
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

    # here get the age information from the file names
    labels_3_classes = {
        "ASTROCYTOMA": 0,
        "EPENDYMOMA": 1,
        "MEDULLOBLASTOMA": 2,
    }
    train_x = [
        int(os.path.basename(f).split("_")[3][0:-1])
        for f in per_fold_training_files[cv_f]
    ]
    train_y = [
        labels_3_classes[os.path.basename(f).split("_")[0]]
        for f in per_fold_training_files[cv_f]
    ]

    val_x = [
        int(os.path.basename(f).split("_")[3][0:-1])
        for f in per_fold_validation_files[cv_f]
    ]
    val_y = [
        labels_3_classes[os.path.basename(f).split("_")[0]]
        for f in per_fold_validation_files[cv_f]
    ]

    test_x = [int(os.path.basename(f).split("_")[3][0:-1]) for f in test_files]
    test_y = [labels_3_classes[os.path.basename(f).split("_")[0]] for f in test_files]

    if args_dict["DATA_NORMALIZATION"]:
        print(" Normalizing age using train set values.")
        # get age norm values from the training set
        mean_age, std_age = np.mean(train_x), np.std(train_x)
        # normalize values
        train_x = (train_x - mean_age) / std_age
        val_x = (val_x - mean_age) / std_age
        test_x = (test_x - mean_age) / std_age

    # make label to categorical
    train_y = to_categorical(train_y, num_classes=args_dict["NBR_CLASSES"])
    val_y = to_categorical(val_y, num_classes=args_dict["NBR_CLASSES"])
    test_y = to_categorical(test_y, num_classes=args_dict["NBR_CLASSES"])

    print(f'{" "*6}Training gen. done!')
    train_gen = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_gen = train_gen.shuffle(1000).repeat().batch(args_dict["BATCH_SIZE"])
    train_steps = np.ceil(len(train_x) / args_dict["BATCH_SIZE"])
    print(f'{" "*6}Validation gen. done!')
    val_gen = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    val_gen = val_gen.batch(args_dict["BATCH_SIZE"])
    val_steps = np.ceil(len(val_gen) / args_dict["BATCH_SIZE"])
    print(f'{" "*6}Testing gen. done!')
    test_gen = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_gen = test_gen.batch(args_dict["BATCH_SIZE"])
    test_steps = np.ceil(len(test_gen) / args_dict["BATCH_SIZE"])

    # -------------
    # CREATE MODEL
    # ------------
    importlib.reload(models)
    print(f'{" "*3}Building model...')
    # build custom model
    model = models.age_only_model(
        args_dict["NBR_CLASSES"], args_dict["MODEL_VERSION"], debug=True
    )

    # ################################# COMPILE MODEL
    # learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    #     args_dict["LEARNING_RATE"], args_dict["MAX_EPOCHS"], 0, power=0.99
    # )

    # values obtained using the LRFind callback

    # learning_rate_fn = tfa.optimizers.CyclicalLearningRate(
    #     initial_learning_rate=1e-2,
    #     maximal_learning_rate=1e0,
    #     scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
    #     step_size=2 * train_steps,
    # )

    learning_rate_fn = None

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

    importlib.reload(tf_callbacks)
    callbacks_list = [
        tf_callbacks.SaveBestModelWeights(
            save_path=best_model_path, monitor="val_loss", mode="min"
        ),
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
    """
    Here evaluate last and best models on both the validation data and the test data
    """
    for m in ["last", "best"]:
        if m == "best":
            model.load_weights(os.path.join(best_model_path, "best_model"))
        for data_gen, data_gen_steps, dataset_name in zip(
            [test_gen, val_gen], [test_steps, val_steps], ["test", "validation"]
        ):
            print(f'{" "*6}Evaluationg {m} model on {dataset_name} data.')
            # get predicitons
            Ptest_softmax = []
            Ytest_categorical = []
            ds_iter = iter(data_gen)
            ds_steps = data_gen_steps
            for i in range(int(ds_steps)):
                x, y = next(ds_iter)
                Ytest_categorical.append(y)
                Ptest_softmax.append(model.predict(x, verbose=0))
            Ptest_softmax = np.row_stack(Ptest_softmax)

            Ptest = np.argmax(Ptest_softmax, axis=-1)
            Ytest_categorical = np.row_stack(Ytest_categorical)

            # save metrics for later if looking at the test set
            if dataset_name == "test":
                summary_test[str(cv_f + 1)][m] = utilities.get_performance_metrics(
                    Ytest_categorical, Ptest_softmax, average="macro"
                )
                summary_test[str(cv_f + 1)][m]["per_case_prediction"] = Ptest

            # plot metrics
            utilities.plotConfusionMatrix(
                GT=Ytest_categorical,
                PRED=Ptest_softmax,
                classes=["Not_tumor", "Tumor"]
                if args_dict["NBR_CLASSES"] == 2
                else (
                    ["ASTR", "EP", "MED"]
                    if args_dict["NBR_CLASSES"] == 3
                    else ["ASTR_in", "ASTR_su", "EP_in", "EP_su", "MED_in"]
                ),
                savePath=save_model_path,
                saveName=f"CM_{m}_model_{dataset_name}",
                draw=False,
            )
            utilities.plotROC(
                GT=Ytest_categorical,
                PRED=Ptest_softmax,
                classes=["Not_tumor", "Tumor"]
                if args_dict["NBR_CLASSES"] == 2
                else (
                    ["ASTR", "EP", "MED"]
                    if args_dict["NBR_CLASSES"] == 3
                    else ["ASTR_in", "ASTR_su", "EP_in", "EP_su", "MED_in"]
                ),
                savePath=save_model_path,
                saveName=f"ROC_{m}_model_{dataset_name}",
                draw=False,
            )
            utilities.plotPR(
                GT=Ytest_categorical,
                PRED=Ptest_softmax,
                classes=["Not_tumor", "Tumor"]
                if args_dict["NBR_CLASSES"] == 2
                else (
                    ["ASTR", "EP", "MED"]
                    if args_dict["NBR_CLASSES"] == 3
                    else ["ASTR_in", "ASTR_su", "EP_in", "EP_su", "MED_in"]
                ),
                savePath=save_model_path,
                saveName=f"PR_{m}_model_{dataset_name}",
                draw=False,
            )

            # save per-subject classification (only test)
            if all([dataset_name == "test", args_dict["NBR_CLASSES"] != 2]):
                MAP_pred = []
                MODE_pred = []
                weighted_MAP_pred = []
                weighted_MODE_pred = []
                for idx, test_subject in enumerate(subj_test_idx):
                    print(f"Working on subject {idx+1}/{len(subj_test_idx)} \r", end="")
                    # get all the files for this subject
                    subj_files = [
                        f for f in test_files if test_subject[0] in os.path.basename(f)
                    ]

                    t_x = [
                        int(os.path.basename(f).split("_")[3][0:-1]) for f in subj_files
                    ]
                    t_x = (t_x - mean_age) / std_age
                    t_y = [
                        labels_3_classes[os.path.basename(f).split("_")[0]]
                        for f in subj_files
                    ]
                    t_y = to_categorical(t_y, num_classes=args_dict["NBR_CLASSES"])
                    # build generator
                    t_gen = tf.data.Dataset.from_tensor_slices((t_x, t_y))
                    t_gen = t_gen.batch(1)
                    # predict cases
                    pred = model.predict(t_gen, verbose=0)
                    # compute aggregated prediction (not weighted)
                    MAP_pred.append(pred.mean(axis=0).argmax())
                    vals, counts = np.unique(pred.argmax(axis=1), return_counts=True)
                    mode_value = np.argwhere(counts == np.max(counts))
                    MODE_pred.append(vals[mode_value].flatten().tolist()[0])

                    # compute aggregated prediction (weighted)
                    slice_position = np.array(
                        [
                            float(Path(os.path.basename(f)).stem.split("_")[-1])
                            for f in subj_files
                        ]
                    )
                    weights = np.where(
                        slice_position < 50, slice_position, 100 - slice_position
                    )
                    # normalize weights (so that the sum goes to 2 (two halfs of the tumor))
                    weights = weights / weights.sum() * 2

                    weighted_MAP_pred.append(
                        (pred * weights[:, np.newaxis]).mean(axis=0).argmax()
                    )

                    # here get the class which sum of weights is the highest
                    weighted_prediction = [
                        np.sum(weights, where=pred.argmax(axis=1) == i)
                        for i in range(args_dict["NBR_CLASSES"])
                    ]
                    weighted_MODE_pred.append(np.argmax(weighted_prediction))

                gt = [i[1] for i in subj_test_idx]
                for pred, pred_name in zip(
                    [MAP_pred, MODE_pred, weighted_MAP_pred, weighted_MODE_pred],
                    ["MeanArgmax", "ArgmaxMode", "wMeanArgmax", "wArgmaxMode"],
                ):
                    utilities.plotConfusionMatrix(
                        GT=gt,
                        PRED=pred,
                        classes=["Not_tumor", "Tumor"]
                        if args_dict["NBR_CLASSES"] == 2
                        else (
                            ["ASTR", "EP", "MED"]
                            if args_dict["NBR_CLASSES"] == 3
                            else ["ASTR_in", "ASTR_su", "EP_in", "EP_su", "MED_in"]
                        ),
                        savePath=save_model_path,
                        saveName=f"CM_{m}_model_{dataset_name}_{pred_name}",
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
        f'Test accuracy -> (last)  {summary_test[str(cv_f+1)]["last"]["overall_accuracy"]:0.3f}, (best) {summary_test[str(cv_f+1)]["best"]["overall_accuracy"]:0.3f}'
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
            f'Test MCC -> (last)  {summary_test[str(cv_f+1)]["last"]["matthews_correlation_coefficient"]:0.3f}, (best) {summary_test[str(cv_f+1)]["best"]["matthews_correlation_coefficient"]:0.3f}'
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
