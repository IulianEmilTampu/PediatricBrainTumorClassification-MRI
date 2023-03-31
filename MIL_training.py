# %%
"""
Script that trains an attention based bag-aggregation methods to classify pediatric brain tumors based on the encoding produced by the SDM4 model.
Conde inspired by the https://arxiv.org/abs/1802.04712 and https://keras.io/examples/vision/attention_mil_classification/

STEPS
- Load dataset information (using splitting from the training of the SDM4 model to avoid bias)
- load the trained SDM4 model (used for feature extraction)
- create one bag for each of the subjects from the training and validation spit using the instances for each subjects: 
 - load the image
 - get encoded vector
 - return bag with encoded indtances
- Build MIL-attention based model
 - shared encoder of the encodigns
 - attention layer
 - aggredation 
 - prediction
- visualize the most important instances in the bag
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import glob
import csv
import shutil
import json
import numpy as np
import argparse
import importlib
import logging
import random
from pathlib import Path
from distutils.util import strtobool
import pandas as pd

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
        "-mr_modelities",
        "--MR_MODALITIES",
        nargs="+",
        required=False,
        default=["T2"],
        help="Specify which MR modalities to use during training (T1 and/or T2)",
    )

    parser.add_argument(
        "-gpu",
        "--GPU_NBR",
        default=0,
        type=str,
        help="Provide the GPU number to use for training.",
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
        "-path_to_encoder_model",
        "--PATH_TO_ENCODER_MODEL",
        required=False,
        type=str,
        default=None,
        help="Specify the path to the encoder model.",
    )
    parser.add_argument(
        "-mil_use_age",
        "--MIL_USE_AGE",
        required=False,
        dest="USE_AGE",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Specify if the MIL model should use the age information. If true, the age information is encoded using a fuly connected model and feature fusion is used to combine image and age infromation.",
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
        "-loss",
        "--LOSS",
        required=False,
        type=str,
        default="CCE",
        help="Specify loss to use during model training (categorical cross entropy CCE, MCC, binary categorical cross entropy BCE. Other can be defined and used. Just implement.",
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

    args_dict = dict(vars(parser.parse_args()))

else:
    if os.name == "posix":
        args_dict = {
            "WORKING_FOLDER": "/flush/iulta54/Research/P5-MICCAI2023",
            "IMG_DATASET_FOLDER": "/flush/iulta54/Research/Data/CBTN/EXTRACTED_SLICES_TFR_MERGED_FROM_TB_20230320",
            "MR_MODALITIES": ["T2"],
            "GPU_NBR": "0",
            "NBR_FOLDS": 5,
            "LEARNING_RATE": 0.0001,
            "BATCH_SIZE": 32,
            "MAX_EPOCHS": 100,
            "PATH_TO_ENCODER_MODEL": "/flush/iulta54/Research/P5-MICCAI2023/trained_models_archive/DetectionModels/DetectionModel_SDM4_t1_t2_BraTS_fullDataset_lr10em6_more_data/fold_1/last_model",
            "PATH_TO_CONFIGURATION_FILES": "/flush/iulta54/Research/P5-MICCAI2023/trained_models_archive/TEST_OVERSAMPLING_EP_with_AUG_optm_ADAM_SDM4_TFRdata_True_modality_T2_loss_MCC_and_CCE_Loss_lr_0.0001_batchSize_32_pretrained_False_frozenWeight_True_useAge_False_simple_age_encoder_useGradCAM_False_seed_1111",
            "MIL_USE_AGE": False,
            "AGE_NORMALIZATION": False,
            "LOSS": "MCC_and_CCE_Loss",
            "RANDOM_SEED_NUMBER": 1111,
            "MODEL_NAME": "MIL_TEST_SDM4_detection",
            "OPTIMIZER": "ADAM",
            "USE_IMAGENET_MODEL": True,
            "IMAGENT_MODEL_NAME": "EfficientNet",
            "BAG_SETTING_BAG_SIZE": None,
            "BAG_SETTING_SORT_BY_SLICE": True,
            "BAG_SETTING_SHUFFLE_BAG": False,
            "MIL_SETTINGS_SHARED_MIL_WEIGHT_SIZE": 64,
        }
    else:
        args_dict = {
            "WORKING_FOLDER": r"C:\Users\iulta54\Documents\PediatricBrainTumorClassification",
            "IMG_DATASET_FOLDER": r"C:\Datasets\CBTN\EXTRACTED_SLICES_TFR_MERGED_FROM_TB_20230320",
            "MR_MODALITIES": ["T2"],
            "GPU_NBR": "0",
            "NBR_FOLDS": 1,
            "LEARNING_RATE": 0.0001,
            "BATCH_SIZE": 32,
            "MAX_EPOCHS": 50,
            "PATH_TO_ENCODER_MODEL": r"C:\Users\iulta54\Documents\PediatricBrainTumorClassification\trained_models_archive\SDM4\fold_2\last_model\last_model",
            "PATH_TO_CONFIGURATION_FILES": r"C:\Users\iulta54\Documents\PediatricBrainTumorClassification\trained_models_archive\SDM4",
            "MIL_USE_AGE": True,
            "AGE_NORMALIZATION": True,
            "LOSS": "MCC_and_CCE_Loss",
            "RANDOM_SEED_NUMBER": 1111,
            "MR_MODALITIES": ["T2"],
            "DEBUG_DATASET_FRACTION": 1,
            "MODEL_NAME": "MIL_TEST",
            "OPTIMIZER": "ADAM",
        }

# revise model name
args_dict[
    "MODEL_NAME"
] = f'{args_dict["MODEL_NAME"]}_optm_{args_dict["OPTIMIZER"]}_loss_{args_dict["LOSS"]}_lr_{args_dict["LEARNING_RATE"]}_batchSize_{args_dict["BATCH_SIZE"]}_useAge_{args_dict["MIL_USE_AGE"]}_seed_{args_dict["RANDOM_SEED_NUMBER"]}'

# --------------------------------------
# set GPU (or device)
# --------------------------------------

# import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
        args_dict["PATH_TO_ENCODER_MODEL"],
    ],
    ["working folder", "image dataset folder", "encoder model"],
):
    if not any([os.path.isdir(folder), os.path.isfile(folder)]):
        raise ValueError(f"{fd.capitalize} not found. Given {folder}.")

# -------------------------------------
# Create folder where to save the model
# -------------------------------------
args_dict["SAVE_PATH"] = os.path.join(
    args_dict["WORKING_FOLDER"], "trained_models_archive", args_dict["MODEL_NAME"]
)
Path(args_dict["SAVE_PATH"]).mkdir(parents=True, exist_ok=True)

# here save the train_validation_test.csv and the config.csv file for this model
shutil.copyfile(
    os.path.join(args_dict["PATH_TO_CONFIGURATION_FILES"], "train_val_test_files.json"),
    os.path.join(args_dict["SAVE_PATH"], "train_val_test_files.json"),
)

# open also the configuration file of the original (per slice classification) training
with open(
    os.path.join(args_dict["PATH_TO_CONFIGURATION_FILES"], "config.json")
) as file:
    config = json.load(file)

# copy in the args_dict some of the values so that the old config can be removed
for key in ["USE_GRADCAM", "USE_AGE", "NBR_CLASSES"]:
    args_dict[key] = config[key]

# print input variables
max_len = max([len(key) for key in args_dict])
[
    print(f"{key:{max_len}s}: {value} ({type(value)})")
    for key, value in args_dict.items()
]

# %% local utilities


def get_per_subject_files_from_csv_file(
    img_dataset_folder, path_to_train_val_test_csv_file, fold
):
    """
    Siple utility that gets the path to the dataset and the path to the train_validation_test.csv file
    and returns the per subject dictionary that contains in each key the files for each subject
    """

    # open .csv file
    with open(path_to_train_val_test_csv_file) as file:
        train_val_test_files = json.load(file)

    per_subject_train_val_test = []
    for ds in ["train", "validation", "test"]:

        # build file names for each subjects
        if ds != "test":
            image_files = [
                os.path.join(
                    args_dict["IMG_DATASET_FOLDER"], args_dict["MR_MODALITIES"][0], f
                )
                for f in train_val_test_files[ds][fold]
            ]
        else:
            image_files = [
                os.path.join(
                    args_dict["IMG_DATASET_FOLDER"], args_dict["MR_MODALITIES"][0], f
                )
                for f in train_val_test_files[ds]
            ]

        # get per subject files
        per_subjects_files = dict.fromkeys(
            [os.path.basename(f).split("_")[2] for f in image_files]
        )
        for subj in per_subjects_files:
            per_subjects_files[subj] = [
                f for f in image_files if subj == os.path.basename(f).split("_")[2]
            ]

        # save
        per_subject_train_val_test.append(per_subjects_files)

    return (
        per_subject_train_val_test[0],
        per_subject_train_val_test[1],
        per_subject_train_val_test[2],
    )


def oversample_class(
    bags,
    bags_labels,
    bags_imgs,
    class_index,
    nbr_oversampling_samples: int = 50,
    random_seed: int = None,
):
    """
    Utility that returns an oversampled version of the given bags with nbr_oversampling_samples more samples
    of the class_index class (0=ASTROCYTOMAS, 1=EPENDYMOMAS, 2=MEDULLOBLASTOMAS)
    """

    print(f"Oversampling class {class_index} with {nbr_oversampling_samples} samples.")
    if random_seed:
        random.seed(random_seed)

    class_samples = np.where(bags_labels.argmax(-1) == class_index)[0]
    random_class_samples = random.choices(class_samples, k=nbr_oversampling_samples)

    # create dummy lists
    aus_bags = list(np.swapaxes(bags, 0, 1))
    aus_bags_labels_bags = list(bags_labels)
    aus_bags_imgs = list(bags_imgs)

    # create dummy lists
    [aus_bags.append(list(np.swapaxes(bags, 0, 1))[i]) for i in random_class_samples]
    [aus_bags_labels_bags.append(list(bags_labels)[i]) for i in random_class_samples]
    [aus_bags_imgs.append(bags_imgs[i]) for i in random_class_samples]

    # put all back into the right shape and/or type
    bags = list(np.swapaxes(aus_bags, 0, 1))
    bags_labels = np.array(aus_bags_labels_bags)
    bags_imgs = aus_bags_imgs

    return bags, bags_labels, bags_imgs


def plot(
    bags_labels,
    bags_images,
    bags_predictions=None,
    bags_attention_weights=None,
    nbr_bags_to_plot: int = 2,
    nbr_imgs_per_bag: int = 3,
    save_image_path: str = None,
    save_name: str = "Example_bag_images",
):

    """ "Utility for plotting bags and attention weights.

    Args:
      bags_encodigs: Input data that contains the bags of instances.
      bags_labels: The associated bag labels of the input data.
      bags_images : The images used to get the encodings (list with in each element a np.array [images, width, higth])
      bags_predictions: Class labels model predictions.
        If you don't specify anything, ground truth labels will be used.
      bags_attention_weights: Attention weights for each instance within the input data.
        If you don't specify anything, the values won't be displayed.
    """

    for b in range(nbr_bags_to_plot):
        # chose random bag from the available ones
        b_idx = np.random.randint(len(bags_images))
        # build figure with nbr_imgs_per_bag images from the randomly selected bag
        fig, ax = plt.subplots(
            nrows=1, ncols=nbr_imgs_per_bag, figsize=(5 * nbr_imgs_per_bag, 7)
        )
        v_min, v_max = bags_images[b_idx].min(), bags_images[b_idx].max()
        for i in range(nbr_imgs_per_bag):
            img = bags_images[b_idx][i]
            if img.shape[-1] == 3:
                img = img[:, :, 0]

            ax[i].imshow(
                img,
                cmap="gray",
                interpolation=None,
                vmin=v_min,
                vmax=v_max,
            )
            if bags_attention_weights is not None:
                ax[i].set_title(
                    f"Attention w.: {bags_attention_weights[b_idx][i]:0.3f}",
                    fontsize=20,
                )
            # make axis pretty
            ax[i].axis("off")
        if bags_predictions is not None:
            plt.suptitle(
                f"Bag nbr. {b_idx+1}\nGT: {bags_labels[b_idx]}\nPred: {bags_predictions[b_idx]}",
                fontsize=20,
            )
        else:
            plt.suptitle(f"Bag nbr. {b_idx+1}\nGT:  {bags_labels[b_idx]}", fontsize=20)

        if save_image_path:
            fig.savefig(os.path.join(save_image_path, f"{save_name}_{b}.png"))
            plt.close(fig)
        else:
            plt.show(fig)


def predict(data, labels, trained_models):

    # Collect info per model.
    models_predictions = []
    models_attention_weights = []

    for model in trained_models:

        # Predict output classes on data.
        predictions = model.predict(data, verbose=0)
        models_predictions.append(predictions)

        # Create intermediate model to get MIL attention layer weights.
        intermediate_model = keras.Model(model.input, model.get_layer("alpha").output)

        # Predict MIL attention layer weights.
        intermediate_predictions = intermediate_model.predict(data, verbose=0)

        attention_weights = np.squeeze(np.swapaxes(intermediate_predictions, 1, 0))
        models_attention_weights.append(attention_weights)

        model.evaluate(data, labels, verbose=0)

    return (
        np.sum(models_predictions, axis=0) / len(trained_models),
        np.sum(models_attention_weights, axis=0) / len(trained_models),
    )


# %% LOAD MODEL
if args_dict["USE_IMAGENET_MODEL"]:
    # load pretrained model from tensorflow
    if args_dict["IMAGENT_MODEL_NAME"] == "EfficientNet":
        print("Loading EfficientnNet model pre-trained on ImagNet")

        img_input = layers.Input(shape=(224, 224, 3), name="image")
        efficientNet = tf.keras.applications.efficientnet.EfficientNetB7(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_tensor=img_input,
        )
        enc_model = tf.keras.Model(inputs=img_input, outputs=efficientNet.output)
    else:
        raise ValueError(
            f"The type of ImageNet pretrained model is not among the ones supported. Add here (easy to implement) :)"
        )
else:
    print("Loading model trained for CBTN slice classification")

    loaded_enc_model = tf.keras.models.load_model(args_dict["PATH_TO_ENCODER_MODEL"])
    # refine model to only get the image encoding vector
    # get index to the global average pooling layer (but using the output of the batch norm that follows the global average pooling)
    idx = [
        i
        for i, l in enumerate(loaded_enc_model.layers)
        if "global_average_pooling2d" in l.name
    ][0]
    img_input = loaded_enc_model.inputs
    encoded_image = loaded_enc_model.layers[idx + 1].output
    enc_model = tf.keras.Model(inputs=img_input, outputs=encoded_image)

# %%
# ---------
# RUNIING CROSS VALIDATION TRAINING
# ---------
importlib.reload(data_utilities)
importlib.reload(models)

# create dictionary where to save the test performance
summary_test = {}
list_of_cv_best_models = []

for cv_f in range(args_dict["NBR_FOLDS"]):
    # make forder where to save the model
    save_model_path = os.path.join(args_dict["SAVE_PATH"], f"fold_{cv_f+1}")
    Path(save_model_path).mkdir(parents=True, exist_ok=True)
    summary_test[str(cv_f + 1)] = {
        "best": {"validation": [], "test": []},
        "last": {"validation": [], "test": []},
    }

    print(f'{" "*3}Setting up training validation and test data ...')

    # -------------------------
    # CREATE DATA GENERATORS
    # -------------------------

    # get per subject paths
    (
        tr_per_subjects_files,
        val_per_subjects_files,
        test_per_subjects_files,
    ) = get_per_subject_files_from_csv_file(
        img_dataset_folder=args_dict["IMG_DATASET_FOLDER"],
        path_to_train_val_test_csv_file=os.path.join(
            args_dict["PATH_TO_CONFIGURATION_FILES"], "train_val_test_files.json"
        ),
        fold=cv_f,
    )

    # encode subjects
    if args_dict["BAG_SETTING_BAG_SIZE"] is None:
        args_dict["BAG_SETTING_BAG_SIZE"] = int(
            np.mean([len(s) for s in tr_per_subjects_files.values()]) / 2
        )

    # Training data
    (
        train_bags,
        train_bags_labels,
        train_bags_imgs,
    ) = data_utilities.get_train_data_MIL_model(
        per_subject_files_dict=tr_per_subjects_files,
        sample_encoder_model=enc_model,
        data_gen_configuration_dict=config,
        bag_size=args_dict["BAG_SETTING_BAG_SIZE"],
        sort_by_slice_location=args_dict["BAG_SETTING_SORT_BY_SLICE"],
        shuffle_samples_in_bag=args_dict["BAG_SETTING_SHUFFLE_BAG"],
        debug_number_of_bags=None,
        rnd_seed=args_dict["RANDOM_SEED_NUMBER"],
        imageNet_pretrained_encoder=args_dict["USE_IMAGENET_MODEL"],
    )

    # Validation data
    (
        val_bags,
        val_bags_labels,
        val_bags_imgs,
    ) = data_utilities.get_train_data_MIL_model(
        per_subject_files_dict=val_per_subjects_files,
        sample_encoder_model=enc_model,
        data_gen_configuration_dict=config,
        bag_size=args_dict["BAG_SETTING_BAG_SIZE"],
        sort_by_slice_location=args_dict["BAG_SETTING_SORT_BY_SLICE"],
        shuffle_samples_in_bag=args_dict["BAG_SETTING_SHUFFLE_BAG"],
        debug_number_of_bags=None,
        rnd_seed=args_dict["RANDOM_SEED_NUMBER"],
        imageNet_pretrained_encoder=args_dict["USE_IMAGENET_MODEL"],
    )

    # Testing data
    if cv_f == 0:
        (
            test_bags,
            test_bags_labels,
            test_bags_imgs,
        ) = data_utilities.get_train_data_MIL_model(
            per_subject_files_dict=test_per_subjects_files,
            sample_encoder_model=enc_model,
            data_gen_configuration_dict=config,
            bag_size=args_dict["BAG_SETTING_BAG_SIZE"],
            sort_by_slice_location=args_dict["BAG_SETTING_SORT_BY_SLICE"],
            shuffle_samples_in_bag=args_dict["BAG_SETTING_SHUFFLE_BAG"],
            debug_number_of_bags=None,
            rnd_seed=args_dict["RANDOM_SEED_NUMBER"],
            imageNet_pretrained_encoder=args_dict["USE_IMAGENET_MODEL"],
        )

    # compute class weights
    class_weights_values = list(
        np.sum(np.bincount(np.array(train_bags_labels).argmax(axis=-1)))
        / (
            len(np.bincount(np.array(train_bags_labels).argmax(axis=-1)))
            * np.bincount(np.array(train_bags_labels).argmax(axis=-1))
        )
    )

    args_dict["CLASS_WEIGHTS"] = {}
    if not all([config["NBR_CLASSES"] == 2, config["DATASET_TYPE"] == "BRATS"]):
        for c in range(config["NBR_CLASSES"]):
            args_dict["CLASS_WEIGHTS"][c] = class_weights_values[c] ** 2
    else:
        for c in range(args_dict["NBR_CLASSES"]):
            args_dict["CLASS_WEIGHTS"][c] = 1

    # OVERSAMPLE EPs
    train_bags, train_bags_labels, train_bags_imgs = oversample_class(
        train_bags,
        train_bags_labels,
        train_bags_imgs,
        class_index=1,
        nbr_oversampling_samples=50,
    )

    # print dataset information
    for bags, bags_labels, dataset_name in zip(
        [train_bags, val_bags, test_bags],
        [train_bags_labels, val_bags_labels, test_bags_labels],
        ["train", "validation", "test"],
    ):
        print("¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤")
        print(f"Dataset: {dataset_name}")
        print(f"    Overall shape of {dataset_name} bags: {np.array(bags).shape}")
        print(f"    Type of {dataset_name} sample: {type(bags[0])}")
        print(
            f"    Overall shape of {dataset_name} label: {np.array(bags_labels).shape}"
        )
        print(f"    Type of {dataset_name} label: {type(bags_labels[0])}")

    # print also class weights
    print(f'Class weights: {args_dict["CLASS_WEIGHTS"]}')

    # -------------------------
    # BUILD MIL MODEL
    # -------------------------

    model = models.MIL_model(
        num_classes=train_bags_labels[0].shape[-1],
        instance_shape=train_bags[0][0].shape[-1],
        bag_size=args_dict["BAG_SETTING_BAG_SIZE"],
        shared_MIL_encoding_dim=args_dict["MIL_SETTINGS_SHARED_MIL_WEIGHT_SIZE"],
    )

    # -------------------------
    # COMPILE MIL MODEL
    # -------------------------

    learning_rate_fn = tfa.optimizers.CyclicalLearningRate(
        initial_learning_rate=1e-3,
        maximal_learning_rate=1e-1,
        scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
        step_size=2 * len(train_bags),
    )

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
            num_classes=config["NBR_CLASSES"]
        )
    elif args_dict["LOSS"] == "MCC_and_CCE_Loss":
        print(f'{" "*6}Using sum of MCC and CCE loss.')
        importlib.reload(losses)
        loss = losses.MCC_and_CCE_Loss()
        what_to_monitor = tfa.metrics.MatthewsCorrelationCoefficient(
            num_classes=config["NBR_CLASSES"]
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
                num_classes=config["NBR_CLASSES"]
            ),
        ],
    )

    # -------------------------
    # GET READY FOR TRAINING
    # -------------------------

    best_model_path = os.path.join(save_model_path, "best_model", "")
    Path(best_model_path).mkdir(parents=True, exist_ok=True)

    importlib.reload(tf_callbacks)
    callbacks_list = [
        tf_callbacks.SaveBestModelWeights(
            save_path=best_model_path, monitor="val_loss", mode="min"
        ),
        # model_checkpoint_callback,
        tf_callbacks.LossAndErrorPrintingCallback(
            save_path=save_model_path, print_every_n_epoch=15
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

    # -------------------------
    # TRAIN MODEL
    # -------------------------

    history = model.fit(
        train_bags,
        train_bags_labels,
        validation_data=(val_bags, val_bags_labels),
        epochs=args_dict["MAX_EPOCHS"],
        batch_size=1,
        callbacks=callbacks_list,
        verbose=1,
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

    # save last training curves
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
        ax[2].legend()

    fig.savefig(os.path.join(save_model_path, "training_curves.png"))
    plt.close(fig)

    # ---------------
    # EVALUATE MODEL
    # ---------------

    for mv in ["last", "best"]:
        # load best model weights
        if mv == "best":
            model.load_weights(os.path.join(best_model_path, "best_model"))

        for bags, bags_labels, bags_imgs, ds_name in zip(
            [val_bags, test_bags],
            [val_bags_labels, test_bags_labels],
            [val_bags_imgs, test_bags_imgs],
            ["validation", "test"],
        ):
            print(f"Running evaluation on {mv} model, {ds_name} dataset...")

            class_predictions, attention_params = predict(bags, bags_labels, [model])
            utilities.plotConfusionMatrix(
                GT=np.array(bags_labels),
                PRED=class_predictions,
                classes=["Not_tumor", "Tumor"]
                if config["NBR_CLASSES"] == 2
                else (
                    ["ASTR", "EP", "MED"]
                    if config["NBR_CLASSES"] == 3
                    else ["ASTR_in", "ASTR_su", "EP_in", "EP_su", "MED_in"]
                ),
                savePath=save_model_path,
                saveName=f"CM_{mv}_model_{ds_name}_fold_{cv_f+1}",
                draw=False,
            )
            utilities.plotROC(
                GT=bags_labels,
                PRED=class_predictions,
                classes=["Not_tumor", "Tumor"]
                if config["NBR_CLASSES"] == 2
                else (
                    ["ASTR", "EP", "MED"]
                    if config["NBR_CLASSES"] == 3
                    else ["ASTR_in", "ASTR_su", "EP_in", "EP_su", "MED_in"]
                ),
                savePath=save_model_path,
                saveName=f"ROC_{mv}_model_{ds_name}_fold_{cv_f+1}",
                draw=False,
            )
            utilities.plotPR(
                GT=bags_labels,
                PRED=class_predictions,
                classes=["Not_tumor", "Tumor"]
                if config["NBR_CLASSES"] == 2
                else (
                    ["ASTR", "EP", "MED"]
                    if config["NBR_CLASSES"] == 3
                    else ["ASTR_in", "ASTR_su", "EP_in", "EP_su", "MED_in"]
                ),
                savePath=save_model_path,
                saveName=f"PR_{mv}_model_{ds_name}_fold_{cv_f+1}",
                draw=False,
            )

            # get metrics
            summary_test[str(cv_f + 1)][mv][
                ds_name
            ] = utilities.get_performance_metrics(
                bags_labels, class_predictions, average="macro"
            )
            # summary_test[str(cv_f + 1)][mv][
            #     "folds_test_logits_values"
            # ] = class_predictions

            # plot some examples
            plot(
                bags_labels=bags_labels,
                bags_images=bags_imgs,
                bags_predictions=class_predictions,
                bags_attention_weights=attention_params,
                nbr_bags_to_plot=5,
                nbr_imgs_per_bag=8,
                save_image_path=save_model_path,
                save_name=f"Example_images_{ds_name}",
            )
# %%
# save all information on file
summary_file = os.path.join(args_dict["SAVE_PATH"], f"tabular_test_summary.csv")
csv_file = open(summary_file, "w")
writer = csv.writer(csv_file)
csv_header = [
    "classification_type",
    "nbr_classes",
    "model_type",
    "model_version",
    "fold",
    "dataset",
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
for k, v in summary_test.items():
    for m in ["last", "best"]:
        for ds_name in ["validation", "test"]:
            try:
                csv_rows.append(
                    [
                        ["Not_tumor", "Tumor"]
                        if config["NBR_CLASSES"] == 2
                        else (
                            ["ASTR", "EP", "MED"]
                            if config["NBR_CLASSES"] == 3
                            else ["ASTR_in", "ASTR_su", "EP_in", "EP_su", "MED_in"]
                        ),
                        config["NBR_CLASSES"],
                        m,
                        "SDM4",
                        ds_name,
                        k,
                        v[m][ds_name]["overall_precision"],
                        v[m][ds_name]["overall_recall"],
                        v[m][ds_name]["overall_accuracy"],
                        v[m][ds_name]["overall_f1-score"],
                        v[m][ds_name]["overall_auc"],
                        v[m][ds_name]["matthews_correlation_coefficient"],
                    ]
                )
            except:
                print("")
writer.writerows(csv_rows)
csv_file.close()
