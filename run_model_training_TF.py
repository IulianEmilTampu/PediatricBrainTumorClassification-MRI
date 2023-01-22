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

# local imports
import utilities
import data_utilities
import models

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
    parser.add_argument(
        "-use_pretrained",
        "--USE_PRETRAINED_MODEL",
        required=False,
        type=bool,
        default=False,
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
        type=bool,
        default=False,
        help="Specify if the model should use the agen information. If true, the age information is encoded using a fuly connected model and feature fusion is used to combine image and age infromation.",
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
        "IMG_DATASET_FOLDER": "/flush/iulta54/Research/Data/CBTN/EXTRACTED_SLICES",
        "DATASET_TYPE": "CBTN",
        "GPU_NBR": "0",
        "MODEL_NAME": "DetectionModel_SDM4_pretrained_BRATS_t1_t2_CBTN_lr10em4_batch_16",
        "NBR_FOLDS": 5,
        "LEARNING_RATE": 0.0001,
        "BATCH_SIZE": 16,
        "MAX_EPOCHS": 50,
        "USE_PRETRAINED_MODEL": True,
        "PATH_TO_PRETRAINED_MODEL": "/flush/iulta54/Research/P5-MICCAI2023/trained_models_archive/SDM4_t2_BraTS_fullDataset_lr10em6_more_data/fold_1/last_model",
        "USE_AGE": False,
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


def get_img_file_names(img_dataset_path: str, dataset_type: str, **kwargs):
    """
    Utility that gets the filenames of the images in the dataset along with the
    unique subject IDs on which to apply the splitting.
    The heuristics on how the filenames and the subjects IDs are obtained depends
    on the dataset type, especially on how the images are storred and named.
    This implementation can handle, for now, the BraTS dataset saved as slices
    and the CBTN dataset saved as slices.

    BraTS dataset structure:
    - modality folder (t1, t1ce, t2, flair)
        - slices_without_tumor_label_0 or slices_with_tumor_label_1
            - slice name (BraTS20_Training_*subjectID*_*modality*_rlp_*relativePositionWrtTheTumor*_label_*label*.jpeg)
    CBTN dataset structure:
    - modality folder (t1, t2)
        - *DIAGNOSIS*_*subjectID*_*scanID*_B_brain_*modality*_rlp_relativePositionWrtTheTumor_label_*0 or 1*.png

    INPUT
    img_dataset_path : str
        Path to where the folders for each modality are located
    dataset_type : str
        Which dataset do the images belong to. This defines the heuristic on how to get the file names
    **kwargs : optional argumets
        modalities : list
            List of modalities to include
        task : str
            Tumor detection (detection) or tumor tupe classification (classification). Specifies if the images with
            no tumor shoul be included (detection) or excluded (classification)
        tumor_min_rpl : int
            Specifies the minimul relative position acceptable for inclusion in the case of a slice with tumor.
            Value between 0 and 100, with 50 indicating the center of the tumor.
        tumor_max_rpl : int
            Specifies the maximum relative position acceptable for inclusion in the case of a slice with tumor
            Value between 0 and 100, with 50 indicating the center of the tumor.
        brain_min_rpl : int
            Specifies the minimul relative position w.r.t the tumor acceptable for inclusion in the case of a slice without tumor.
            Value between 1 and n, with n indicating the number of slices away from the closes tumor slice.
        brain_max_rpl : int
            Specifies the maximum relative position w.r.t the tumor acceptable for inclusion in the case of a slice without tumor.
            Value between 1 and n, with n indicating the number of slices away from the closes tumor slice.

    OUTPUT
    all_file_names : list of tuples
        Where the first element if the path to the file and the second element is the subjectID the imege belongs to
    """

    # check the optional arguments and initialize them if not present
    # optional arguments
    default_opt_args = {
        "modalities": [
            os.path.basename(f) for f in glob.glob(os.path.join(img_dataset_path, "*"))
        ],
        "task": "detection",
        "tumor_min_rpl": 0,
        "tumor_max_rpl": 100,
        "brain_min_rpl": 1,
        "brain_max_rpl": 25,
    }

    if not kwargs:
        kwargs = default_opt_args
    else:
        for keys, values in default_opt_args.items():
            if keys not in kwargs.keys():
                kwargs[keys] = values

    print(kwargs)

    # initiate variables to be returned
    all_file_names = []

    # work on each dataset type
    if dataset_type.upper() == "BRATS":
        for modality in kwargs["modalities"]:
            # specify if to include slices without tumor
            if kwargs["task"].lower() == "detection":
                classes = ["slices_without_tumor_label_0", "slices_with_tumor_label_1"]
            elif kwargs["task"].lower() == "classification":
                classes = ["slices_with_tumor_label_1"]

            for s in classes:
                files = glob.glob(os.path.join(img_dataset_path, modality, s, "*.jpeg"))
                if s == "slices_with_tumor_label_1":
                    # filter files with relative position of the tumor
                    files = [
                        (f, int(os.path.basename(f).split("_")[2]))
                        for f in files
                        if all(
                            [
                                float(os.path.basename(f).split("_")[5])
                                >= kwargs["tumor_min_rpl"],
                                float(os.path.basename(f).split("_")[5])
                                <= kwargs["tumor_max_rpl"],
                            ]
                        )
                    ]
                if s == "slices_without_tumor_label_0":
                    # filter files with relative position of the tumor within [20, 80]
                    files = [
                        (f, int(os.path.basename(f).split("_")[2]))
                        for f in files
                        if all(
                            [
                                float(os.path.basename(f).split("_")[5])
                                >= kwargs["brain_min_rpl"],
                                float(os.path.basename(f).split("_")[5])
                                <= kwargs["brain_max_rpl"],
                            ]
                        )
                    ]
                all_file_names.extend(files)

    if dataset_type.upper() == "CBTN":
        for modality in kwargs["modalities"]:
            # work on the images with tumor
            files = glob.glob(os.path.join(img_dataset_path, modality, "*label_1.png"))
            # filter files with relative position of the tumor
            files = [
                (f, os.path.basename(f).split("_")[1])
                for f in files
                if all(
                    [
                        float(os.path.basename(f).split("_")[-3])
                        >= kwargs["tumor_min_rpl"],
                        float(os.path.basename(f).split("_")[-3])
                        <= kwargs["tumor_max_rpl"],
                    ]
                )
            ]
            all_file_names.extend(files)
            # add the files without tumor if detection
            if kwargs["task"].lower() == "detection":
                files = glob.glob(
                    os.path.join(img_dataset_path, modality, "*label_0.png")
                )
                files = [
                    (f, os.path.basename(f).split("_")[1])
                    for f in files
                    if all(
                        [
                            float(os.path.basename(f).split("_")[-3])
                            >= kwargs["brain_min_rpl"],
                            float(os.path.basename(f).split("_")[-3])
                            <= kwargs["brain_max_rpl"],
                        ]
                    )
                ]
                all_file_names.extend(files)

    return all_file_names


# # ######### this it the default on
# # # get the unique subjects in the IMG_DATASET_FOLDER
# # # NOTE that the heuristic used to get the unique patient IDs might be changed depending on the dataset type

all_file_names = get_img_file_names(
    img_dataset_path=args_dict["IMG_DATASET_FOLDER"],
    dataset_type="CBTN",
    modalities=["T2"],
    task="detection",
    tumor_min_rpl=0,
    tumor_max_rpl=100,
    brain_min_rpl=1,
    brain_max_rpl=25,
)

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
# %% test genrators
importlib.reload(data_utilities)
target_size = (224, 224)
gen = data_utilities.get_data_generator_TF_CBTN(
    sample_files=test_files,
    target_size=target_size,
    batch_size=args_dict["BATCH_SIZE"],
    dataset_type="training",
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
    # train_gen = data_utilities.get_data_generator_TF_CBTN(
    #     sample_files=per_fold_training_files[cv_f],
    #     target_size=target_size,
    #     batch_size=args_dict["BATCH_SIZE"],
    #     dataset_type="training",
    # )
    # val_gen = data_utilities.get_data_generator_TF_CBTN(
    #     sample_files=per_fold_validation_files[cv_f],
    #     target_size=target_size,
    #     batch_size=args_dict["BATCH_SIZE"],
    #     dataset_type="validation",
    # )
    # test_gen = data_utilities.get_data_generator_TF_CBTN(
    #     sample_files=test_files,
    #     target_size=target_size,
    #     batch_size=args_dict["BATCH_SIZE"],
    #     dataset_type="testing",
    # )

    print(
        f"Training: {len(train_gen)}\nValidation: {len(val_gen)}\nTesting: {len(test_gen)}"
    )

    ## BUILD DETERCTION MODEL
    importlib.reload(models)
    if args_dict["USE_PRETRAINED_MODEL"]:
        print(f'{" "*3}Loading pretrained model...')
        # load model
        model = tf.keras.models.load_model(args_dict["PATH_TO_PRETRAINED_MODEL"])
        # replace the last dense layer to match the number of classes
        intermediat_output = model.layers[-2].output
        new_output = tf.keras.layers.Dense(
            units=3, input_shape=model.layers[-1].input_shape, name="prediction"
        )(intermediat_output)
        model = tf.keras.Model(inputs=model.inputs, outputs=new_output)
    else:
        print(f'{" "*3}Building model from scratch...')
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

    loss = tf.keras.losses.CategoricalCrossentropy()

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
