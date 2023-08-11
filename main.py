import os
from pathlib import Path
import argparse
import json
from distutils.util import strtobool
from typing import Any, Union, Sequence
from datetime import datetime
import logging
import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("agg")

from filelock import FileLock
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import tensorflow_addons as tfa


import ray
from ray import air, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.search import ConcurrencyLimiter

# local imports
import data_utilities
import models
import losses
import tf_callbacks


def parse_recipes() -> dict:
    parser = argparse.ArgumentParser(
        description="Running model training for classification of pediatric tumors."
    )
    parser.add_argument("--recipe_file", default=None, required=False)

    # general settings
    parser.add_argument(
        "--working_folder",
        required=False,
        type=str,
        help="Provide the working folder where the trained model will be saved.",
    )
    parser.add_argument(
        "--model_name",
        required=False,
        type=str,
        default="MICCAI2023",
        help="Name used to save the model and the scores",
    )
    parser.add_argument(
        "--script_running_mode",
        choices=(
            "hyper_parameter_tuning",
            "training",
            "evaluation",
            "inference",
        ),
        required=False,
        type=str,
        default="train",
        help="Specify if the script runs model hyper_parameter_optimization, training, evaluation or inference.",
    )

    # data loader settings
    parser.add_argument(
        "--dataset_path",
        required=False,
        type=str,
        help="Provide the Image Dataset Folder where the folders for each modality are located (see dataset specifications in the README file).",
    )
    parser.add_argument(
        "--dataset_type",
        required=False,
        default="CBTN",
        type=str,
        help="Provide the image dataset type (BRATS, CBTN, CUSTOM). This will set the dataloader appropriate for the dataset.",
    )
    parser.add_argument(
        "--mr_modelities",
        nargs="+",
        required=False,
        default=["T2"],
        help="Specify which MR modalities to use during training (T1 and/or T2)",
    )
    parser.add_argument(
        "--trf_data",
        required=False,
        dest="TFR_DATA",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Specify if the dataset used for training originates from TFRecord files. This is used to choose between data generators.",
    )
    parser.add_argument(
        "--num_classes",
        required=False,
        type=int,
        default=2,
        help="Number of classification classes.",
    )
    parser.add_argument(
        "--num_folds",
        required=False,
        type=int,
        default=1,
        help="Number of cross validation folds.",
    )
    parser.add_argument(
        "--data_augmentation",
        required=False,
        default=True,
        type=bool,
        help="Specify if data augmentation should be applied",
    )
    parser.add_argument(
        "--data_normalization",
        required=False,
        default=True,
        type=bool,
        help="Specify if data normalization should be performed (check pretrained models if they need it)",
    )
    parser.add_argument(
        "--data_scale",
        required=False,
        default=False,
        type=bool,
        help="Specify if data should be scaled (from [0,255] to [0,1]) (check pretrained models if they need it)",
    )
    parser.add_argument(
        "--use_age",
        required=False,
        dest="USE_AGE",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Specify if the model should use the age information. If true, the age information is encoded using a fuly connected model and feature fusion is used to combine image and age infromation.",
    )
    parser.add_argument(
        "--age_normalization",
        required=False,
        dest="AGE_NORMALIZATION",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Specify if the age should be normalized. If True, age is normalized using mean and std from the trianing dataset ([-1,1] norm).",
    )
    parser.add_argument(
        "--use_gradCAM",
        required=False,
        dest="USE_GRADCAM",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Specify if the model should use the gradCAM information. If true, the gradCAM infromation is concatenated to the image information as an extra channel",
    )
    parser.add_argument(
        "--target_size",
        nargs="+",
        required=False,
        default=[224, 224],
        help="Specify target size of the images for model training.",
    )

    # training settings
    parser.add_argument(
        "--gpu_nbr",
        default=0,
        type=str,
        help="Provide the GPU number to use for training.",
    )
    parser.add_argument(
        "--learning_rate",
        required=False,
        type=float,
        default=0.0001,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        required=False,
        type=int,
        default=16,
        help="Specify batch size. Default 16",
    )
    parser.add_argument(
        "-max_epochs",
        required=False,
        type=int,
        default=300,
        help="Number of max training epochs.",
    )
    parser.add_argument(
        "--optimizer",
        required=False,
        type=str,
        default="SGD",
        help="Specify which optimizer to use. Here one can set SGD or ADAM. Others can be implemented.",
    )
    parser.add_argument(
        "--loss",
        required=False,
        type=str,
        default="CCE",
        help="Specify loss to use during model training (categorical cross entropy CCE, MCC, binary categorical cross entropy BCE. Other can be defined and used. Just implement.",
    )
    parser.add_argument(
        "--random_seed_number",
        required=False,
        type=int,
        default=29122009,
        help="Specify random number seed. Useful to have models trained and tested on the same data.",
    )

    # model settings
    parser.add_argument(
        "--model_type",
        required=False,
        type=str,
        default="SDM4",
        help="Specify model to use during training. Chose among the ones available in the models.py file.",
    )
    parser.add_argument(
        "--use_pretrained_model",
        required=False,
        dest="use_pretrained_model",
        type=lambda x: bool(strtobool(x)),
        help="Specify if the image encoder should be loading the weight pretrained on BraTS",
    )
    parser.add_argument(
        "--path_to_pretrained_model",
        required=False,
        type=str,
        default=None,
        help="Specify the path to the pretrained model to use as image encoder.",
    )
    parser.add_argument(
        "--freeze_weights",
        required=False,
        dest="FREEZE_WEIGHTS",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Specify if pretrained model weights should be frozen.",
    )
    parser.add_argument(
        "--age_encoder_version",
        required=False,
        type=str,
        default="no_encoder",
        help="Available age encoders: no_encoder | simple_age_encode | large_age_encoder",
    )

    # debug settings
    parser.add_argument(
        "--debug_dataset_fraction",
        required=False,
        type=float,
        default=1.0,
        help="Specify the percentage of the dataset to use during training and validation. This is for debug",
    )
    parser.add_argument(
        "--debug_print_batch_examples",
        required=False,
        type=bool,
        default=False,
        help="Specify if example images from the batched data generator should be saved in the Working folder (under Example_batched_images).",
    )

    recipe_file_arg, recipe_inline_args = parser.parse_known_args()

    if recipe_file_arg.recipe_file is not None:
        # parse reciep file.
        # TODO add check of the different variables based on a template!
        if os.path.isfile(recipe_file_arg.recipe_file):
            print("Reciep file provided. Parsing variables...")
            with open(recipe_file_arg.recipe_file) as json_file:
                recipe = json.load(json_file)
        else:
            raise ValueError(
                f"The recipe file provided does not exist. Provide a valid file. Given {recipe_file_arg.recipe_file}"
            )
        # fix the type of the values in the recipe
        recipe["dataloader_settings"]["data_normalization"] = (
            True
            if recipe["dataloader_settings"]["data_normalization"] == "True"
            else False
        )
        recipe["dataloader_settings"]["data_agumentation"] = (
            True
            if recipe["dataloader_settings"]["data_agumentation"] == "True"
            else False
        )
        recipe["dataloader_settings"]["data_scale"] = (
            True if recipe["dataloader_settings"]["data_scale"] == "True" else False
        )
        recipe["dataloader_settings"]["tfr_data"] = (
            True if recipe["dataloader_settings"]["tfr_data"] == "True" else False
        )
        recipe["dataloader_settings"]["use_age"] = (
            True if recipe["dataloader_settings"]["use_age"] == "True" else False
        )
        recipe["dataloader_settings"]["age_normalization"] = (
            True
            if recipe["dataloader_settings"]["age_normalization"] == "True"
            else False
        )
        recipe["dataloader_settings"]["use_gradCAM"] = (
            True if recipe["dataloader_settings"]["use_gradCAM"] == "True" else False
        )

        recipe["model_settings"]["use_pretrained"] = (
            True if recipe["model_settings"]["use_pretrained"] == "True" else False
        )
        recipe["model_settings"]["freeze_weights"] = (
            True if recipe["model_settings"]["freeze_weights"] == "True" else False
        )

        recipe["debug_settings"]["debug_print_batch_examples"] = (
            True
            if recipe["debug_settings"]["debug_print_batch_examples"] == "True"
            else False
        )

    else:
        print("Parsing settings...")
        # manually parse the remaining values creating a dictionary matching the recipe template one
        recipe = _get_default_recipe(with_comments=False)
        # polulate dictionary with the provided data.
        # TODO define message errors
        recipe["working_folder"] = recipe_inline_args.working_folder
        recipe["script_running_mode"] = recipe_inline_args.script_running_mode
        recipe["run_name"] = recipe_inline_args.run_name

        recipe["dataloader_settings"][
            "dataset_folder"
        ] = recipe_inline_args.dataset_folder
        recipe["dataloader_settings"]["dataset_type"] = recipe_inline_args.dataset_type
        recipe["dataloader_settings"][
            "data_normalization"
        ] = recipe_inline_args.data_normalization
        recipe["dataloader_settings"][
            "data_agumentation"
        ] = recipe_inline_args.data_agumentation
        recipe["dataloader_settings"]["data_scale"] = recipe_inline_args.data_scale
        recipe["dataloader_settings"]["tfr_data"] = recipe_inline_args.tfr_data
        recipe["dataloader_settings"]["use_age"] = recipe_inline_args.use_age
        recipe["dataloader_settings"][
            "age_normalization"
        ] = recipe_inline_args.age_normalization
        recipe["dataloader_settings"]["use_gradCAM"] = recipe_inline_args.use_gradCAM
        recipe["dataloader_settings"][
            "mr_modelities"
        ] = recipe_inline_args.mr_modelities
        recipe["dataloader_settings"]["num_classes"] = recipe_inline_args.num_classes
        recipe["dataloader_settings"]["num_folds"] = recipe_inline_args.num_folds

        recipe["model_settings"]["model_type"] = recipe_inline_args.model_type
        recipe["model_settings"]["use_pretrained"] = recipe_inline_args.use_pretrained
        recipe["model_settings"][
            "path_to_pretrained_model"
        ] = recipe_inline_args.path_to_pretrained_model
        recipe["model_settings"]["freeze_weights"] = recipe_inline_args.freeze_weights
        recipe["model_settings"][
            "age_encoder_version"
        ] = recipe_inline_args.age_encoder_version
        recipe["model_settings"][
            "hyper_parameters_recipe"
        ] = recipe_inline_args.hyper_parameters_recipe

        recipe["training_settings"]["gpu_nbr"] = recipe_inline_args.gpu_nbr
        recipe["training_settings"]["learning_rate"] = recipe_inline_args.learning_rate
        recipe["training_settings"]["batch_size"] = recipe_inline_args.batch_size
        recipe["training_settings"]["max_epochs"] = recipe_inline_args.max_epochs
        recipe["training_settings"]["loss"] = recipe_inline_args.loss
        recipe["training_settings"]["optimizer"] = recipe_inline_args.optimizer
        recipe["training_settings"][
            "random_seed_number"
        ] = recipe_inline_args.random_seed_number

        recipe["debug_settings"][
            "debug_dataset_fraction"
        ] = recipe_inline_args.debug_dataset_fraction
        recipe["debug_settings"][
            "debug_print_batch_examples"
        ] = recipe_inline_args.debug_print_batch_examples

    # check name of the run. Add date and time
    date_time = datetime.now()
    d = date_time.strftime("t%H%M_d%m%d%Y")
    recipe["run_name"] = "_".join(
        [
            # d,
            recipe["run_name"],
            recipe["model_settings"]["model_type"],
            "optm",
            recipe["training_settings"]["optimizer"],
            "tfr_data",
            str(recipe["dataloader_settings"]["tfr_data"]),
            # "modality",
            # "_".join([i for i in recipe["dataloader_settings"]["mr_modelities"]]),
            "loss",
            recipe["training_settings"]["loss"],
            "lr",
            str(recipe["training_settings"]["learning_rate"]),
            "batchSize",
            str(recipe["training_settings"]["batch_size"]),
            "pretrained",
            str(recipe["model_settings"]["use_pretrained"]),
            "frozenWeights",
            str(recipe["model_settings"]["freeze_weights"]),
            "use_age",
            str(recipe["dataloader_settings"]["use_age"]),
            "age_encoder_version",
            recipe["model_settings"]["age_encoder_version"],
            "rnd_seed",
            str(recipe["training_settings"]["random_seed_number"]),
        ]
    )
    print("Done. Here is a summary of the settings:")
    # print_dict.pd(recipe)

    return recipe


def set_up(recipe):
    """
    Utility that give the recipe, sets the visible GPU for training and created the path to where the script run is saved
    """

    # -------------------------------------
    # Create folder where to save the model
    # -------------------------------------
    recipe["SAVE_PATH"] = os.path.join(
        recipe["working_folder"], "trained_models_archive", recipe["run_name"]
    )
    Path(recipe["SAVE_PATH"]).mkdir(parents=True, exist_ok=True)

    # -------------------------------------
    # Start logger in the SAVE_PATH folder
    # -------------------------------------
    logging.basicConfig(
        filename=os.path.join(recipe["SAVE_PATH"], "logs.txt"),
        level=logging.DEBUG,
        format="pedMRI - %(asctime)s - %(message)s",
    )
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)

    # define logger for this application
    setup_logger = logging.getLogger("main.setup")

    setup_logger.info(
        f' ### STARTING JOB WITH SCRIPT IN {recipe["script_running_mode"].capitalize()} ### '
    )
    setup_logger.info(f'Saving job information at {recipe["SAVE_PATH"]}')

    # --------------------------------------
    # set GPU (or device)
    # --------------------------------------

    # import tensorflow
    # import tensorflow as tf
    # import tensorflow_addons as tfa
    # import warnings

    # tf.get_logger().setLevel(logging.ERROR)
    # from tensorflow.keras.utils import to_categorical
    # from tensorflow_addons.optimizers import Lookahead

    # # from tensorflow.python.framework.ops import disable_eager_execution

    # # disable_eager_execution()

    # devices = tf.config.list_physical_devices("GPU")

    # if devices:
    #     setup_logger.info(
    #         f'Running training on GPU # {recipe["training_settings"]["gpu_nbr"]} \n'
    #     )

    #     warnings.simplefilter(action="ignore", category=FutureWarning)
    #     tf.config.experimental.set_memory_growth(devices[0], True)
    # else:
    #     setup_logger.warning(
    #         f"ATTENTION!!! MODEL RUNNING ON CPU. Check implementation in case GPU is wanted."
    #     )

    # -------------------------------------
    # Check that the given folder exist
    # -------------------------------------
    for folder, fd in zip(
        [
            recipe["working_folder"],
            recipe["dataloader_settings"]["dataset_folder"],
        ],
        [
            "working folder",
            "image dataset folder",
        ],
    ):
        if not os.path.isdir(folder):
            raise ValueError(f"{fd.capitalize} not found. Given {folder}.")


def _get_default_recipe(with_comments=True):
    """
    Utility that returns an empty recipe that contains all the different fiends that can
    be set and run using this repository.
    If with_comments=True, comments for each entry are provided, else each is empty (useful for actually putting in the values).
    """
    recipe = {
        "working_folder": "Provide the working folder where the trained model will be saved.",
        "script_running_mode": "",
        "run_name": "Name used to save the model and the scores",
        "dataloader_settings": {
            "dataset_folder": "Provide the Image Dataset Folder where the folders for each modality are located (see dataset specifications in the README file).",
            "dataset_type": "Provide the image dataset type (BRATS, CBTN, CUSTOM). This will set the dataloader appropriate for the dataset.",
            "data_normalization": "Specify if data normalization should be performed (check pretrained models if they need it)",
            "data_agumentation": "Specify if data augmentation should be applied",
            "data_scale": "Specify if data should be scaled (from [0,255] to [0,1]) (check pretrained models if they need it)",
            "tfr_data": "Specify if the dataset used for training originates from TFRecord files. This is used to choose between data generators.",
            "use_age": "pecify if the model should use the age information. If true, the age information is encoded using a fuly connected model and feature fusion is used to combine image and age infromation.",
            "age_normalization": "Specify if the age should be normalized. If True, age is normalized using mean and std from the trianing dataset ([-1,1] norm).",
            "use_gradCAM": "Specify if the model should use the gradCAM information. If true, the gradCAM infromation is concatenated to the image information as an extra channel",
            "mr_modelities": "Specify which MR modalities to use during training (T1 and/or T2)",
            "num_classes": "Number of classification classes.",
            "num_folds": "Number of cross validation folds.",
            "target_size": "Specify the size of the input images to the model. DEfault [224,224].",
        },
        "model_setting": {
            "model_type": "Specify model to use during training. Chose among the ones available in the models.py file.",
            "use_pretrained": "pecify if the image encoder should be loading the weight pretrained on BraTS",
            "path_to_pretrained_model": "Specify the path to the pretrained model to use as image encoder.",
            "freeze_weights": "Specify if pretrained model weights should be frozen.",
            "age_encoder_version": "Available age encoders: no_encoder | simple_age_encode | large_age_encoder",
            "hyper_parameters_recipe": "Path to the json file with the model hyper_parameters fund using this script when running in hyper_parameter_tuning mode.",
        },
        "training_settings": {
            "gpu_nbr": "Provide the GPU number to use for training.",
            "learning_rate": "Learning rate",
            "batch_size": "Specify batch size.",
            "max_epochs": "Number of max training epochs.",
            "loss": "pecify loss to use during model training (categorical cross entropy CCE, MCC, binary categorical cross entropy BCE. Other can be defined and used. Just implement.",
            "optimizer": "Specify which optimizer to use. Here one can set SGD or ADAM. Others can be implemented.",
            "random_seed_number": "This controlls the random splitting of the data in the different folds. This can be used to getrepeated cross-validations",
        },
        "debug_settings": {
            "debug_dataset_fraction": "Specify the percentage of the dataset to use during training and validation. This is for debug",
            "debug_print_batch_examples": "Specify if example images from the batched data generator should be saved in the Working folder (under Example_batched_images).",
        },
    }

    if with_comments:
        for key, value in recipe.values():
            if isinstance(value, dict):
                for kkey, vvalue in value.items():
                    recipe[key][kkey] = ""
            else:
                recipe[key] = ""

    return recipe


def _get_dataset_files_for_training_validation_testing(recipe):
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.model_selection import KFold
    from sklearn.utils.extmath import softmax

    """
    Independently from the combination of modalities, the test validation and train sets
    are defined so that no vlomume is present in more than one set.

    Steps
    2 - using the number of subject, screate indexes to identify which files
        are used for training, validation and testing
    3 - save the information about the split.
    """
    logger = logging.getLogger("main.get_dataset_files")
    logger.info("Splitting dataset (per-volume (subject) splitting).")

    # get all the file names. Here one can filter among which files to take for this job.
    all_file_names = data_utilities.get_img_file_names(
        img_dataset_path=recipe["dataloader_settings"]["dataset_folder"],
        dataset_type=recipe["dataloader_settings"]["dataset_type"],
        modalities=recipe["dataloader_settings"]["mr_modelities"],
        return_labels=True,
        task="detection"
        if recipe["dataloader_settings"]["num_classes"] == 2
        else "classification",
        nbr_classes=recipe["dataloader_settings"]["num_classes"],
        tumor_min_rpl=10,
        tumor_max_rpl=90,
        brain_min_rpl=1,
        brain_max_rpl=100,
        file_format="tfrecords"
        if recipe["dataloader_settings"]["tfr_data"]
        else "jpeg",
        tumor_loc=["infra", "supra"],
    )
    logger.info("Could get the files to train/validate/test on.")

    # get unique subjects
    unique_patien_IDs_with_labels = list(
        dict.fromkeys([(f[1], f[2]) for f in all_file_names])
    )
    unique_patien_IDs_labels = [f[1] for f in unique_patien_IDs_with_labels]

    # save information
    recipe["in_script_settings"] = {
        "number_of_available_subjects": len(unique_patien_IDs_with_labels)
    }

    # get stratified split
    subj_train_val_idx, subj_test_idx = train_test_split(
        unique_patien_IDs_with_labels,
        stratify=unique_patien_IDs_labels
        if not all(
            [
                recipe["dataloader_settings"]["num_classes"] == 2,
                recipe["dataloader_settings"]["dataset_type"] == "BRATS",
            ]
        )
        else None,
        test_size=0.20,
        random_state=recipe["training_settings"]["random_seed_number"],
    )

    # get actual test files
    test_files = [
        f[0] for f in all_file_names if any([i[0] == f[1] for i in subj_test_idx])
    ]
    logger.info(f'{"# Train-val subjects":18s}: {len(subj_train_val_idx):2d}')
    logger.info(
        f'{"# Test subjects":18s}: {len(subj_test_idx):2d} ({len(test_files)} total images)'
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

    logger.info("Calculating class weights in the training/validation set.")
    recipe["in_script_settings"]["class_weights"] = {}
    if not all(
        [
            recipe["dataloader_settings"]["num_classes"] == 2,
            recipe["dataloader_settings"]["dataset_type"] == "BRATS",
        ]
    ):
        for c in range(recipe["dataloader_settings"]["num_classes"]):
            recipe["in_script_settings"]["class_weights"][c] = (
                class_weights_values[c] ** 2
            )
    else:
        for c in range(recipe["dataloader_settings"]["num_classes"]):
            recipe["in_script_settings"]["class_weights"] = 1

    # # OVERSAMPLING THE EP class
    # random.seed(args_dict["RANDOM_SEED_NUMBER"])
    # EP_samples = [i for i in subj_train_val_idx if i[1] == 1]
    # subj_train_val_idx.extend(random.choices(EP_samples, k=50))
    # subj_train_val_idx_labels = [f[1] for f in subj_train_val_idx]

    subj_train_idx, subj_val_idx = [], []
    per_fold_training_files, per_fold_validation_files = [], []
    # set cross validation
    if recipe["dataloader_settings"]["num_folds"] > 1:
        kf = StratifiedKFold(
            n_splits=recipe["dataloader_settings"]["num_folds"],
            shuffle=True,
            random_state=recipe["training_settings"]["random_seed_number"],
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

            # # print to check that all is good
            # logger.info(f'Fold {idx+1}: {""*4}{"training":10s} ->{subj_train_idx[-1]}')
            # logger.info(f'Fold {idx+1}: {""*4}{"validation":10s} ->{subj_val_idx[-1]}')
    else:
        # N_FOLDS is only one, setting 10% of the training dataset as validation
        logger.info("Getting indexes of training and validation files for one fold.")
        aus_train, aus_val = train_test_split(
            subj_train_val_idx,
            stratify=subj_train_val_idx_labels
            if not all(
                [
                    recipe["dataloader_settings"]["num_classes"] == 2,
                    recipe["dataloader_settings"]["dataset_type"] == "BRATS",
                ]
            )
            else None,
            test_size=0.1,
            random_state=recipe["training_settings"]["random_seed_number"],
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
            [
                f[0]
                for f in all_file_names
                if any([i[0] == f[1] for i in subj_val_idx[-1]])
            ]
        )

        # # print to check that all is good
        # logger.info(
        #     f'Fold {recipe["dataloader_settings"]["num_folds"]}: {""*4}{"training":10s} ->{subj_train_idx[-1]} ({len(per_fold_training_files[-1])} images) '
        # )
        # logger.info(
        #     f'Fold {recipe["dataloader_settings"]["num_folds"]}: {""*4}{"validation":10s} ->{subj_val_idx[-1]} ({len(per_fold_validation_files[-1])} images)'
        # )

    # check that no testing files are in the training or validation
    check_test_files = True
    if check_test_files:
        idx_to_remove = []
        remove_overlap = True
        for idx, test_f in enumerate(test_files):
            # print(
            #     f"Checking test files ({idx+1:0{len(str(len(test_files)))}d}\{len(test_files)})\r",
            #     end="",
            # )
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

        logger.info(f"Checking of the test files passed!")
    else:
        logger.info(f"WARNING!!!\nSKipping check of test file.")

    # Save infromation about which files are used for training/validation/testing
    data_dict = {
        "test": [f for f in test_files],
        "train": [],
        "validation": [],
    }

    for f in range(recipe["dataloader_settings"]["num_folds"]):
        data_dict["train"].append([i for i in per_fold_training_files[f]])
        data_dict["validation"].append([i for i in per_fold_validation_files[f]])

    logger.info(f"# Training files:{len(per_fold_training_files[-1])}")
    logger.info(f"# Validation files: {len(per_fold_validation_files[-1])}")

    logger.info("Saving used file names in SAVE_PATH")
    with open(
        os.path.join(recipe["SAVE_PATH"], "train_val_test_files.json"), "w"
    ) as file:
        json.dump(data_dict, file)

    return data_dict


def _get_train_validation_test_data_generators(
    train_set: list,
    validation_set: list,
    recipe: dict,
    test_set=None,
):
    """
    Utility that given the paths to the  training, validation and testing files, returns a data generator on them using the recipe
    information.
    """
    logger = logging.getLogger("main.get_datagenerators")
    if recipe["dataloader_settings"]["tfr_data"]:
        data_gen = data_utilities.tfrs_data_generator
    elif any(
        [
            recipe["dataloader_settings"]["use_age"],
            recipe["dataloader_settings"]["use_gradCAM"],
        ]
    ):
        raise ValueError(
            "Trying to run training using dataset from .jpeg files while asking for age information and/or gradCAM.\nThis is not yet implemented. Use TFR dataset."
        )
    else:
        data_gen = data_utilities.img_data_generator

    # shuffle the training files
    random.Random(recipe["training_settings"]["random_seed_number"]).shuffle(train_set)

    # get norm stat if needed
    if recipe["dataloader_settings"]["data_normalization"]:
        train_gen, train_steps = data_gen(
            file_paths=train_set[
                0 : int(
                    len(train_set) * recipe["debug_settings"]["debug_dataset_fraction"]
                )
            ],
            input_size=recipe["dataloader_settings"]["target_size"],
            batch_size=recipe["training_settings"]["batch_size"],
            buffer_size=1000,
            return_gradCAM=recipe["dataloader_settings"]["use_gradCAM"],
            return_age=recipe["dataloader_settings"]["use_age"],
            dataset_type="test",  # this is just to not have the dataset repeated infinitely (changes after)
            nbr_classes=recipe["dataloader_settings"]["num_classes"],
        )

        # get normalization stats
        logger.info(f"Getting normalization stats from training generator...")
        norm_stats = data_utilities.get_normalization_values(
            train_gen,
            train_steps,
            return_age_norm_values=recipe["dataloader_settings"]["use_age"],
            return_gradCAM_norm_values=recipe["dataloader_settings"]["use_gradCAM"],
        )
    else:
        norm_stats = [None, None, None]

    # build actuall training datagen with normalized values
    output_as_RGB = (
        True
        if any(
            [
                recipe["model_settings"]["model_type"] == "EfficientNet",
                recipe["model_settings"]["model_type"] == "ResNet50",
                recipe["model_settings"]["model_type"] == "VGG16",
            ]
        )
        else False
    )

    train_gen, train_steps = data_gen(
        file_paths=train_set[
            0 : int(len(train_set) * recipe["debug_settings"]["debug_dataset_fraction"])
        ],
        input_size=recipe["dataloader_settings"]["target_size"],
        batch_size=recipe["training_settings"]["batch_size"],
        buffer_size=500,
        return_gradCAM=recipe["dataloader_settings"]["use_gradCAM"],
        return_age=recipe["dataloader_settings"]["use_age"],
        dataset_type="train",
        nbr_classes=recipe["dataloader_settings"]["num_classes"],
        output_as_RGB=output_as_RGB,
    )
    logger.info("Training generator done.")

    random.Random(recipe["training_settings"]["random_seed_number"]).shuffle(
        validation_set
    )
    val_gen, val_steps = data_gen(
        file_paths=validation_set[
            0 : int(
                len(validation_set) * recipe["debug_settings"]["debug_dataset_fraction"]
            )
        ],
        input_size=recipe["dataloader_settings"]["target_size"],
        batch_size=recipe["training_settings"]["batch_size"],
        buffer_size=1000,
        return_gradCAM=recipe["dataloader_settings"]["use_gradCAM"],
        return_age=recipe["dataloader_settings"]["use_age"],
        dataset_type="val",
        nbr_classes=recipe["dataloader_settings"]["num_classes"],
        output_as_RGB=output_as_RGB,
    )
    logger.info("Validation generator done.")

    if test_set:
        # return test set as well (TODO)
        test_gen, test_steps = None, None
        logger.info("Testing generator done.")

        return (
            norm_stats,
            train_gen,
            train_steps,
            val_gen,
            val_steps,
            test_gen,
            test_steps,
        )
    else:
        return norm_stats, train_gen, train_steps, val_gen, val_steps


def _build_model_based_on_recipe(recipe):
    """
    Utility that builds the model as set in the recipe
    """
    logger = logging.getLogger("main.build_model")
    input_shape = (
        (
            recipe["dataloader_settings"]["target_size"][0],
            recipe["dataloader_settings"]["target_size"][1],
            1,
        )
        if not recipe["dataloader_settings"]["use_gradCAM"]
        else (
            recipe["dataloader_settings"]["target_size"][0],
            recipe["dataloader_settings"]["target_size"][1],
            2,
        )
    )

    if recipe["model_settings"]["model_type"] == "SDM4":
        logger.info(f'Using {recipe["model_settings"]["model_type"]} model.')
        return models.SimpleDetectionModel_TF(
            num_classes=recipe["dataloader_settings"]["num_classes"],
            input_shape=input_shape,
            image_normalization_stats=recipe["model_settings"]["norm_stats"][0],
            scale_image=recipe["dataloader_settings"]["data_scale"],
            data_augmentation=recipe["dataloader_settings"]["data_agumentation"],
            kernel_size=(3, 3),
            pool_size=(2, 2),
            use_age=recipe["dataloader_settings"]["use_age"],
            age_normalization_stats=recipe["model_settings"]["norm_stats"][2]
            if recipe["dataloader_settings"]["age_normalization"]
            else None,
            age_encoder_version=recipe["model_settings"]["age_encoder_version"],
            use_pretrained=recipe["model_settings"]["use_pretrained"],
            pretrained_model_path=recipe["model_settings"]["path_to_pretrained_model"],
            freeze_weights=recipe["model_settings"]["freeze_weights"],
            debug=True,
        )

    elif recipe["model_settings"]["model_type"] == "ResNet9":
        logger.info(f'Using {recipe["model_settings"]["model_type"]} model.')
        return models.ResNet9(
            num_classes=recipe["dataloader_settings"]["num_classes"],
            input_shape=input_shape,
            use_age=recipe["dataloader_settings"]["use_age"],
            use_age_thr_tabular_network=False,
        )
    elif recipe["model_settings"]["model_type"] == "ViT":
        logger.info(f'Using {recipe["model_settings"]["model_type"]} model.')
        return models.ViT(
            input_size=input_shape,
            num_classes=recipe["dataloader_settings"]["num_classes"],
            use_age=recipe["dataloader_settings"]["use_age"],
            use_age_thr_tabular_network=False,
            use_gradCAM=recipe["dataloader_settings"]["use_gradCAM"],
            patch_size=16,
            projection_dim=64,
            num_heads=8,
            mlp_head_units=(256, 128),
            transformer_layers=8,
            transformer_units=None,
            debug=False,
        )
    elif recipe["model_settings"]["model_type"] == "EfficientNet":
        logger.info(f'Using {recipe["model_settings"]["model_type"]} model.')
        return models.EfficientNet(
            num_classes=recipe["dataloader_settings"]["num_classes"],
            input_shape=(input_shape[0], input_shape[1], 3),
            use_age=recipe["dataloader_settings"]["use_age"],
            use_age_thr_tabular_network=False,
            pretrained=recipe["model_settings"]["use_pretrained"],
            freeze_weights=recipe["model_settings"]["freeze_weights"],
        )
    elif recipe["model_settings"]["model_type"] == "VGG16":
        logger.info(f'Using {recipe["model_settings"]["model_type"]} model.')
        return models.VGG16(
            num_classes=recipe["dataloader_settings"]["num_classes"],
            input_shape=(input_shape[0], input_shape[1], 3),
            use_age=recipe["dataloader_settings"]["use_age"],
            use_age_thr_tabular_network=False,
            pretrained=recipe["model_settings"]["use_pretrained"],
            freeze_weights=recipe["model_settings"]["freeze_weights"],
            debug=True,
        )
    elif recipe["model_settings"]["model_type"] == "ResNet50":
        logger.info(f'Using {recipe["model_settings"]["model_type"]} model.')
        return models.ResNet50(
            num_classes=recipe["dataloader_settings"]["num_classes"],
            input_shape=(input_shape[0], input_shape[1], 3),
            use_age=recipe["dataloader_settings"]["use_age"],
            use_age_thr_tabular_network=False,
            pretrained=recipe["model_settings"]["use_pretrained"],
            freeze_weights=recipe["model_settings"]["freeze_weights"],
        )
    else:
        raise ValueError(
            "Model type not among the ones that are implemented.\nDefine model in the models.py file and add code here for building the model."
        )


def _train_model(config, recipe=None, num_cross_validation_fold=None):
    data_dict = _get_dataset_files_for_training_validation_testing(recipe)
    (
        norm_stats,
        train_gen,
        train_steps,
        val_gen,
        val_steps,
    ) = _get_train_validation_test_data_generators(
        train_set=data_dict["train"][num_cross_validation_fold],
        validation_set=data_dict["validation"][num_cross_validation_fold],
        recipe=recipe,
        test_set=None,
    )

    recipe["model_settings"]["norm_stats"] = norm_stats

    model = _build_model_based_on_recipe(recipe)

    # specify optimizer
    if config["optimizer"] == "SGD":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=config["learning_rate"], momentum=config["momentum"]
        )
    elif config["optimizer"] == "ADAM":
        optimizer = tfa.optimizers.AdamW(
            learning_rate=config["learning_rate"],
            weight_decay=0.0001,
        )

    # specify loss
    if config["loss"] == "MCC":
        # logger.info(f"Using MCC loss.")
        loss = losses.MCC_Loss()
        what_to_monitor = tfa.metrics.MatthewsCorrelationCoefficient(
            num_classes=recipe["dataloader_settings"]["num_classes"]
        )
    elif config["loss"] == "MCC_and_CCE_Loss":
        # logger.info(f"Using sum of MCC and CCE loss.")
        loss = losses.MCC_and_CCE_Loss()
    elif config["loss"] == "CCE":
        # logger.info(f"Using CCE loss.")
        loss = tf.keras.losses.CategoricalCrossentropy()
    elif config["loss"] == "BCE":
        # logger.info(f"Using BCS loss.")
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        raise ValueError(
            f"The loss provided is not available. Implement in the losses.py or here."
        )

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    # specify callbacks
    callbacks_list = [
        tf_callbacks.LossAndErrorPrintingCallback(
            save_path=os.path.join(
                recipe["SAVE_PATH"], f"fold_{num_cross_validation_fold+1}"
            ),
            print_every_n_epoch=10,
        ),
        tf_callbacks.SaveBestModelWeights(
            save_path=os.path.join(
                recipe["SAVE_PATH"], f"fold_{num_cross_validation_fold+1}"
            ),
            monitor="val_loss",
            mode="min",
        ),
        TuneReportCallback({"mean_accuracy": "val_accuracy"}),
    ]

    model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        batch_size=recipe["training_settings"]["batch_size"],
        epochs=recipe["training_settings"]["max_epochs"],
        verbose=0,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks_list,
    )


def _tune_CBTN(num_training_iterations, num_cross_validation_fold):
    Path(
        os.path.join(recipe["SAVE_PATH"], f"fold_{num_cross_validation_fold+1}")
    ).mkdir(parents=True, exist_ok=True)

    sched = tune.schedulers.ASHAScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20
    )

    # train_fn_with_parameters = tune.with_parameters(train_model, recipe=recipe)

    train_fn_with_parameters = tune.with_parameters(
        _train_model, recipe=recipe, num_cross_validation_fold=num_cross_validation_fold
    )

    tuner = tune.Tuner(
        tune.with_resources(train_fn_with_parameters, resources={"cpu": 15, "gpu": 1}),
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max",
            scheduler=sched,
            num_samples=recipe["tuner_settings"]["num_samples"],
        ),
        run_config=air.RunConfig(
            name=f"fold_{num_cross_validation_fold+1}",
            stop={
                "mean_accuracy": 0.90,
                "training_iteration": num_training_iterations,
            },
            storage_path=os.path.join(recipe["SAVE_PATH"], "ray_tuner"),
        ),
        # param_space={
        #     "learning_rate": tune.uniform(0.001, 0.1),
        #     "momentum": tune.uniform(0.1, 0.9),
        #     "optimizer": tune.choice(["SGD", "ADAM"]),
        #     "loss": tune.choice(["MCC", "CCE", "MCC_and_CCE_Loss"]),
        # },
        param_space={
            "learning_rate": tune.uniform(
                recipe["tuner_settings"]["learning_rate_range"][0],
                recipe["tuner_settings"]["learning_rate_range"][1],
            ),
            "momentum": tune.uniform(
                recipe["tuner_settings"]["momentum_range"][0],
                recipe["tuner_settings"]["momentum_range"][1],
            ),
            "optimizer": tune.choice(recipe["tuner_settings"]["optimizer"]),
            "loss": tune.choice(recipe["tuner_settings"]["loss"]),
        },
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)


def _run_recipe(recipe):
    """
    Function that takes runs the recipe.
    """
    # define logger for this application
    logger = logging.getLogger("main.run_recipe")

    if recipe["script_running_mode"] == "hyper_parameter_tuning":
        logger.info("Running script in hyper parameter tuning mode.")

        for cv_f in range(recipe["dataloader_settings"]["num_folds"]):
            _tune_CBTN(
                num_training_iterations=recipe["training_settings"]["max_epochs"],
                num_cross_validation_fold=cv_f,
            )

    elif recipe["script_running_mode"] == "train":
        logger.info("Running script in training mode.")

        # _run_model_training(recipe)

    elif recipe["script_running_mode"] == "validation":
        logger.info("Running script in validation mode.")
    elif recipe["script_running_mode"] == "inference":
        logger.info("Running script in inference mode.")
    else:
        raise ValueError(
            f'Running script mode on among the ones accepted. Given {recipe["script_running_mode"]}'
        )


if __name__ == "__main__":
    recipe = parse_recipes()
    set_up(recipe)
    _run_recipe(recipe)
