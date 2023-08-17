# %%
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

# %% DEFINE PATHS and SET EXTRACTION PARAMETERS

SAVE_PATH = "/flush/iulta54/Research/Data/CBTN_v1/EXTRACTED_SLICES_JPG"
IMG_DATASET_FOLDER = (
    "/flush/iulta54/Research/Data/CBTN_v1/EXTRACTED_SLICES_TFR_MERGED_FROM_TB_20230320"
)
DATASET_TYPE = "CBTN"
MR_MODALITIES = ["T2"]
NBR_CLASSES = 3
TFR_DATA = True

TUMOR_MIN_RELATIVE_POSITION = 30
TUMOR_MAX_RELATIVE_POSITION = 70

# %% GET FILE NAMES
all_file_names = data_utilities.get_img_file_names(
    img_dataset_path=IMG_DATASET_FOLDER,
    dataset_type=DATASET_TYPE,
    modalities=MR_MODALITIES,
    return_labels=True,
    task="classification",
    nbr_classes=NBR_CLASSES,
    tumor_min_rpl=TUMOR_MIN_RELATIVE_POSITION,
    tumor_max_rpl=TUMOR_MAX_RELATIVE_POSITION,
    brain_min_rpl=1,
    brain_max_rpl=100,
    file_format="tfrecords" if TFR_DATA else "jpeg",
    tumor_loc=["infra", "supra"],
)

# %% SPLIT IN TRAIN VELID AND TEST

# the unique patients IDs
unique_patien_IDs_with_labels = list(
    dict.fromkeys([(f[1], f[2]) for f in all_file_names])
)
unique_patien_IDs_labels = [f[1] for f in unique_patien_IDs_with_labels]

# take out test
subj_train_val_idx, subj_test_idx = train_test_split(
    unique_patien_IDs_with_labels,
    stratify=unique_patien_IDs_labels,
    test_size=0.20,
    random_state=29122009,
)

test_files = [
    f[0] for f in all_file_names if any([i[0] == f[1] for i in subj_test_idx])
]

train_val_subj = [i[0] for i in subj_train_val_idx]
train_val_label = [i[1] for i in subj_train_val_idx]

# get validation
subj_train_idx, subj_val_idx = train_test_split(
    train_val_subj,
    stratify=train_val_label,
    test_size=0.10,
    random_state=29122009,
)

train_files = [f[0] for f in all_file_names if any([i == f[1] for i in subj_train_idx])]
val_files = [f[0] for f in all_file_names if any([i == f[1] for i in subj_val_idx])]

print(
    f"Training files: {len(train_files)}\nValidation files: {len(val_files)}\nTest files: {len(test_files)}"
)


# %% BUILD GENERATOR ON THESE FILE NAMES
from PIL import Image

labels_3_classes = {
    0: "ASTROCYTOMA",
    1: "EPENDYMOMA",
    2: "MEDULLOBLASTOMA",
}

# build folders where to save images
for set_name in ["TRAIN", "VALIDATION", "TEST"]:
    for class_name in labels_3_classes.values():
        Path(
            os.path.join(
                SAVE_PATH
                + f"_{TUMOR_MIN_RELATIVE_POSITION}_{TUMOR_MAX_RELATIVE_POSITION}",
                set_name,
                class_name,
            )
        ).mkdir(parents=True, exist_ok=True)


for set_name, set_files in zip(
    ["TRAIN", "VALIDATION", "TEST"], [train_files, val_files, test_files]
):
    gen, gen_steps = data_utilities.tfrs_data_generator(
        file_paths=set_files,
        input_size=(224, 224),
        batch_size=1,
        buffer_size=1,
        return_gradCAM=False,
        return_age=False,
        dataset_type="test",
        nbr_classes=NBR_CLASSES,
        output_as_RGB=False,
    )

    # SAVE IMAGES
    gen_iter = iter(gen)
    for i in range(gen_steps):
        print(f"Working on {set_name} set ({i+1:04}/{gen_steps})    \r", end="")
        # get images
        img, label = next(gen_iter)
        # save image
        img = img["image"].numpy().squeeze()
        img = Image.fromarray(img)

        # build image name
        save_path = os.path.join(
            SAVE_PATH + f"_{TUMOR_MIN_RELATIVE_POSITION}_{TUMOR_MAX_RELATIVE_POSITION}",
            set_name,
            labels_3_classes[label.numpy().argmax()],
            f"{labels_3_classes[label.numpy().argmax()]}_{i:04d}.jpg",
        )

        # get file name
        img.convert("RGB").save(save_path)
