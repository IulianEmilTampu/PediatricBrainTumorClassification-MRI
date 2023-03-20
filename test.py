# %%
"""
Script that runs model testing over cross validation training.

Steps
1 - get the path to the dataset and the models
2 - build data generator and model
3 - run test for each model
4 - ensamble predictions
5 - save results
"""
import os
import glob
import csv
import json
import numpy as np
import argparse
import importlib
import logging
from pathlib import Path
import time

# local imports
import utilities
import data_utilities

# import trainer

su_debug_flag = True

# --------------------------------------
# read the input arguments and set the base folder
# --------------------------------------
if not su_debug_flag:
    parser = argparse.ArgumentParser(description="Run test on cross validation models")
    parser.add_argument(
        "-wf",
        "--WORKING_FOLDER",
        required=True,
        type=str,
        help="Provide the utilities scripts are located",
    )
    parser.add_argument(
        "-wf",
        "--CV_MODEL_FOLDER",
        required=True,
        type=str,
        help="Provide the path where the cross vlaidation models are saved. This folder must contain the train_val_test_files.json file",
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
    args_dict = dict(vars(parser.parse_args()))

else:
    # # # # # # # # # # # # # # DEBUG
    args_dict = {
        "WORKING_FOLDER": "/flush/iulta54/Research/P5-MICCAI2023",
        "CV_MODEL_FOLDER": "Classification_optm_ADAM_SDM4_TFRdata_True_modality_T2_loss_MCC_and_CCE_Loss_lr_0.0001_batchSize_32_pretrained_False_useAge_False_useGradCAM_False_seed_1111",
        "IMG_DATASET_FOLDER": "/flush/iulta54/Research/Data/CBTN/EXTRACTED_SLICES_TFR_TB_20230309",
        "GPU_NBR": "0",
    }

# --------------------------------------
# set GPU (or device)
# --------------------------------------

# import tensorflow
import tensorflow as tf
import warnings

tf.get_logger().setLevel(logging.ERROR)
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
args_dict["SAVE_PATH"] = os.path.join(args_dict["CV_MODEL_FOLDER"], "test_summary")

Path(args_dict["SAVE_PATH"]).mkdir(parents=True, exist_ok=True)

# print input variables
max_len = max([len(key) for key in args_dict])
[print(f"{key:{max_len}s}: {value}") for key, value in args_dict.items()]

# %% GET TEST FILES AND BUILD GENERATOR

with open(
    os.path.join(args_dict["CV_MODEL_FOLDER"], "train_val_test_files.json")
) as file:
    train_val_test_files = json.load(file)
    test_files = [
        os.path.join(args_dict["IMG_DATASET_FOLDER"], os.path.basename(f))
        for f in train_val_test_files["test"]
    ]

    # get also per subject files
    per_subjects_files = dict.fromkeys(
        [os.path.basename(f).split("-")[2] for f in test_files]
    )
    for subj in per_subjects_files:
        per_subjects_files[subj] = [
            f for f in test_files if subj == os.path.basename(f).split("-")[2]
        ]

# open also the configuration file
with open(os.path.join(args_dict["CV_MODEL_FOLDER"], "config.json")) as file:
    config = json.load(file)

# build generator
target_size = (224, 224)
test_gen, test_steps = data_utilities.tfrs_data_generator(
    file_paths=test_files,
    input_size=target_size,
    batch_size=1,
    buffer_size=10,
    return_gradCAM=config["USE_GRADCAM"],
    return_age=config["USE_AGE"],
    dataset_type="test",
    nbr_classes=config["NBR_CLASSES"],
    output_as_RGB=True
    if any(
        [
            config["MODEL_TYPE"] == "EfficientNet",
            config["MODEL_TYPE"] == "ResNet50",
        ]
    )
    else False,
)

# %% GET PATHS TO THE MODELS TO TEST
# get models to run prediction on (along with the index of the images to use for
# each of them based on the dataset).

MODELS = []
for f_indx, f in enumerate(
    glob.glob(os.path.join(args_dict["CV_MODEL_FOLDER"], "fold_*", ""))
):
    # get paths to best and last model
    aus_dict = {"last": None, "best": None}
    for m, mv, mvn in zip(
        ["last", "best"],
        ["last_model", "best_model_weights"],
        ["last_model", "best_model"],
    ):
        if os.path.isfile(os.path.join(f, mv, mvn)):
            aus_dict[m] = os.path.join(f, mv, mvn)
    MODELS.append(aus_dict)

# print what has been found
for idx, m in enumerate(MODELS):
    print(
        f"Fold {idx+1}\n  Last model: {True if m['last'] else None}\n    Best model: {True if m['best'] else None}"
    )

# %% GET TEST PERFORMANCE ON A SLICE LEVEL

summary_test = {}
for cv_f, M in enumerate(MODELS):
    summary_test[str(cv_f + 1)] = {"best": [], "last": []}
    for mv in ["last", "best"]:
        if M[mv]:
            # load model
            model = tf.keras.models.load_model(M[mv])
            # get predicitons
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

            summary_test[str(cv_f + 1)][mv] = utilities.get_performance_metrics(
                Ytest_categorical, Ptest_softmax, average="macro"
            )
            summary_test[str(cv_f + 1)][mv]["folds_test_logits_values"] = Ptest

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
                savePath=args_dict["SAVE_PATH"],
                saveName=f"CM_{mv}_model_fold_{cv_f+1}",
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
                savePath=args_dict["SAVE_PATH"],
                saveName=f"ROC_{mv}_model_fold_{cv_f+1}",
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
                savePath=args_dict["SAVE_PATH"],
                saveName=f"PR_{mv}_model_fold_{cv_f+1}",
                draw=False,
            )

# plot ensamble prediction
for mv in ["last", "best"]:
    # ############ plot and save confucion matrix
    ensemble_pred_argmax = []
    ensemble_pred_logits = []

    # compute the logits mean along the folds (only concatenate the values that are not None -> models that were not saved)
    ensemble_pred_logits = np.array(
        summary_test[str(cv_f + 1)][mv]["folds_test_logits_values"]
        for cv_f in range(len(MODELS))
        if MODELS[cv_f][mv]
    )
    # compute argmax prediction
    ensemble_pred_argmax = np.argmax(ensemble_pred_logits, axis=1)

    acc = utilities.plotConfusionMatrix(
        Ytest_categorical,
        ensemble_pred_argmax,
        classes=config["label_description"],
        savePath=args_dict["SAVE_PATH"],
        saveName=f"ensemble_CM_{mv}_model",
        draw=False,
    )

    # ############ plot and save ROC curve
    fpr, tpr, roc_auc = utilities.plotROC(
        Ytest_categorical,
        ensemble_pred_logits,
        classes=config["label_description"],
        savePath=args_dict["SAVE_PATH"],
        saveName=f"ensemble_ROC_{mv}_model",
        draw=False,
    )
