"""
Script that creates TFRecord data samples containing CBTN images, with age and gradCAM images included.
"""
# %%
import glob  # Unix style pathname pattern expansion
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import tensorflow as tf
from pathlib import Path
from PIL import Image

## General functions to convert values to a type compatible to a tf.exampe


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# %% GET PATHS
to_print = "    Convert CBTN dataset into TFRecord dataset (image + age + gradCAM)   "

print(f'\n{"-"*len(to_print)}')
print(to_print)
print(f'{"-"*len(to_print)}\n')

su_debug_flag = False

# --------------------------------------
# read the input arguments and set the base folder
# --------------------------------------
if not su_debug_flag:
    parser = argparse.ArgumentParser(
        description="Convert CBTN dataset into TFRecord dataset (image + age + gradCAM)."
    )
    parser.add_argument(
        "-pti",
        "--PATH_TO_CBTN_SLICES",
        required=True,
        help="Pathe to the folder containing the transversal slices of the CBTN dataset to convert to TFRecords",
    )
    parser.add_argument(
        "-ptg",
        "--PATH_TO_CBTN_GRADCAMS",
        required=True,
        type=str,
        help="Path to the .ny GradCAM files for the CBTN slices",
    )
    parser.add_argument(
        "-s",
        "--SAVE_PATH",
        required=True,
        type=str,
        help="Where to save the dataset",
    )

else:
    # # # # # # # # # # # # # # DEBUG
    print("Running in debug mode.")
    args_dict = {
        "PATH_TO_CBTN_SLICES": "/flush/iulta54/Research/Data/CBTN/EXTRACTED_SLICES/T2",
        "PATH_TO_CBTN_GRADCAMS": "/flush/iulta54/Research/P5-MICCAI2023/trained_models_archive/DetectionModels/DetectionModel_SDM4_t2_CBTN_loss_CCE_lr_1e-05_batchSize_32_pretrained_False_useAge_False/Explainability_analysis/GradCAMs",
        "PATH_TO_TABULAR_DATA": "/flush/iulta54/Research/Data/CBTN/CSV_files/t2_t1c_all_files.xlsx",
        "SAVE_PATH": "/flush/iulta54/Research/Data/CBTN/EXTRACTED_SLICES_TFR/T2",
    }

# make save path
Path(args_dict["SAVE_PATH"]).mkdir(parents=True, exist_ok=True)

# print input variables
max_len = max([len(key) for key in args_dict])
[print(f"{key:{max_len}s}: {value}") for key, value in args_dict.items()]

# %%  GET FILES TO CONVERT
img_files = glob.glob(os.path.join(args_dict["PATH_TO_CBTN_SLICES"], "*.png"))
img_files = [f for f in img_files if "annotation" not in os.path.basename(f)]
gcam_files = glob.glob(os.path.join(args_dict["PATH_TO_CBTN_GRADCAMS"], "*.npy"))

# check which files have a gradCAM file
missing_gradCAM = []
img_and_gradCAM_files = []
for img_f in img_files:
    # build gradCAM file name
    aus_f = os.path.join(
        args_dict["PATH_TO_CBTN_GRADCAMS"], Path(os.path.basename(img_f)).stem + ".npy"
    )
    if aus_f not in gcam_files:
        missing_gradCAM.append(Path(os.path.basename(img_f)).stem)
    else:
        img_and_gradCAM_files.append((img_f, aus_f))


print(f"Total number of image files: {len(img_files)}")
print(f"Total number of GradCAM files: {len(gcam_files)}")
print(f"Image files missing the corresponding gradCAM: {len(missing_gradCAM)}")

# %%  OPEN TABULAR FILE WHERE THE LABEL FOR EACH SAMPLE IS STORED

labels_3_classes = {
    "ASTROCYTOMA": 1,
    "EPENDYMOMA": 2,
    "MEDULLOBLASTOMA": 3,
}

labels_5_classes = {
    "ASTROCYTOMAinfra": 1,
    "ASTROCYTOMAsupra": 2,
    "EPENDYMOMAinfra": 3,
    "EPENDYMOMAsupra": 4,
    "MEDULLOBLASTOMAinfra": 5,
}

df_info = pd.read_excel(args_dict["PATH_TO_TABULAR_DATA"])

# %%  CREATE TFRdata SAMPLES

skipped_files = []
save_non_tumor_slices = True
saved_idx = 0
# load image data and gradCAM for each slice
for idx, f in enumerate(img_and_gradCAM_files):
    print(
        f"Working on {idx+1:{len(str(len(img_and_gradCAM_files)))}d} of {len(img_and_gradCAM_files)} (saved {saved_idx+1}) \r",
        end="",
    )
    # get if the file belongs to a slice with or without tumor
    slice_with_tumor = int(Path(os.path.basename(f[0])).stem.split("_")[-1])

    if all([save_non_tumor_slices, not slice_with_tumor]):
        # set labels of slice to None since there is no tumor
        label_3_classes, label_5_classes = 0, 0
    else:
        # get label for 3 classes and 5 classes
        label_3_classes = labels_3_classes[os.path.basename(f[0]).split("_")[0]]
        # for the label for 5 classes, use the df_info file
        try:
            idx_in_df = df_info.index[
                df_info["subject_session"]
                == "_".join(os.path.basename(f[0]).split("_")[1:5])
            ][0]
            label_5_classes = labels_5_classes[df_info.loc[idx_in_df].at["d_l"]]
        except:
            # do not save those slices with tumor which to not have a double classification
            skipped_files.append(f[0])
            continue

    # load image
    if any([slice_with_tumor, ~slice_with_tumor * save_non_tumor_slices]):
        img_data = np.array(Image.open(f[0]), dtype="float32")
        # load gradCAM
        gradCAM_data = np.load(f[1], allow_pickle=True).item()
        gradCAM_data = [
            np.squeeze(gradCAM_data["gradCAM_raw"][m][:, :, 1, 0])
            for m in range(len(gradCAM_data["gradCAM_raw"]))
        ]
        gradCAM_data = np.mean(
            np.stack(gradCAM_data, axis=-1), axis=-1, dtype="float32"
        )
        # get age from file
        age_data = int(os.path.basename(f[0]).split("_")[2][0:-1])

        # get ready to write the tf_example into the TFrecord file
        # add infra supra information in the file name
        file_name = Path(os.path.basename(f[0])).stem
        if any([label_5_classes == i for i in [1, 3, 5]]):
            file_name = (
                file_name.split("_")[0]
                + "_infra_"
                + "_".join(file_name.split("_")[1::])
                + ".tfrecords"
            )
            loc = "infra"
        else:
            file_name = (
                file_name.split("_")[0]
                + "_supra_"
                + "_".join(file_name.split("_")[1::])
                + ".tfrecords"
            )
            loc = "supra"

        writer = tf.io.TFRecordWriter(os.path.join(args_dict["SAVE_PATH"], file_name))

        # Creates a tf.Example message ready to be written to a file for all the images.
        feature = {
            "xdim": _int64_feature(int(img_data.shape[0])),
            "ydim": _int64_feature(int(img_data.shape[1])),
            "age": _int64_feature(age_data),
            "image": _bytes_feature(tf.compat.as_bytes(img_data.tostring())),
            "gradCAM": _bytes_feature(tf.compat.as_bytes(gradCAM_data.tostring())),
            "file_name": _bytes_feature(tf.compat.as_bytes(f[0])),
            "slice_with_tumor": _int64_feature(int(slice_with_tumor)),
            "label_3_classes": _int64_feature(int(label_3_classes)),
            "label_5_classes": _int64_feature(int(label_5_classes)),
            "tumor_location": _bytes_feature(tf.compat.as_bytes(loc)),
        }

        # wrap feature with the Example class
        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
        # write to file
        writer.write(tf_example.SerializeToString())

        saved_idx += 1

        # close file
        writer.close()
        del feature
        del tf_example

# %% CHECK THAT ALL IIS GOOD BY OPENING THE TFR FILE
## DEBUG -> OPEN, READ AND SHOW A SUBJECT
import importlib
import data_utilities
import random

DEBUG = True

if DEBUG == True:
    # GET TFRECORD FILES
    tfr_files = glob.glob(os.path.join(args_dict["SAVE_PATH"], "*.tfrecords"))

    print(f"Found {len(tfr_files)} files.")
    random.shuffle(tfr_files)

    # creating datagen from tfr
    importlib.reload(data_utilities)

    data_gen = data_utilities.data_generator(
        file_paths=tfr_files,
        input_size=(224, 224),
        batch_size=16,
        buffer_size=100,
        data_augmentation=False,
        return_age=False,
        return_gradCAM=False,
        dataset_type="train",
        nbr_classes=2,
    )

    # plot samples (image, gradCAM and age info)

    sample = next(iter(data_gen))
    batch_idx = 0
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    ax[0].imshow(
        np.squeeze(sample[0]["image"].numpy())[batch_idx, :, :],
        cmap="gray",
        interpolation=None,
    )
    try:
        ax[1].imshow(
            np.squeeze(sample[0]["image"].numpy())[batch_idx, :, :, 1],
            cmap="gray",
            interpolation=None,
        )
    except:
        print("No gradCAM found")
    try:
        print(sample[0]["age"][batch_idx])
    except:
        print("No age found")
    print(sample[1]["label"][batch_idx])
