"""
Script that merges TB's TFR dataset into IET's dataset framework.
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


## functions to open TB's TFR files
def parser(
    proto,
    input_size,
):
    """
    This function parses a single TF examples.
    In this implementation, the sample is only parsed and brought back to its
    original shape. Note that other manipulations can be applied if needed
    (not data augmentation, for example addigng extra channels)

    INPUT
    proto : tf.data.TFRecordDataset
        This (internaly) has the link to the path where an TF example is stored.
    input_size : tuple or list
        Size of the image to be parce. Note that the size of the image is also stored
        in the tf example, but given that the code is built into a graph, the size
        of the image can not be left unspecified (the parced input dimentions are only
        tensors which value is defined at runtime - the resize function need values
        available during graph construction).

    OUTPUT
    image : tf eager tensor (has the numpy() attribute)
    """
    # TO MATCH TB's DATASET STRUCTURE
    key_features = {
        "image_data": tf.io.FixedLenFeature([], tf.string),
        "age_days": tf.io.FixedLenFeature([], tf.int64),
        "image_label_3_classes": tf.io.FixedLenFeature([], tf.int64),
        "image_label_5_classes": tf.io.FixedLenFeature([], tf.int64),
    }
    # take out specified features (key_features) from the example
    parsed_features = tf.io.parse_single_example(proto, key_features)

    image = tf.io.decode_raw(parsed_features["image_data"], out_type=tf.float32)
    image = tf.reshape(image, shape=input_size)

    return image


def get_image_from_TFR_file(file_name, input_size: tuple = (224, 224)):
    """
    Utility that given a TFRecord file and a parcer that can open it, returns the numpy array of the iamge
    """
    raw_example = tf.data.TFRecordDataset(file_name)
    parsed_example = raw_example.map(
        tf.autograph.experimental.do_not_convert(
            lambda x: (
                parser(
                    x,
                    input_size,
                )
            )
        )
    )

    for image_features in parsed_example:
        image_raw = image_features.numpy()
    return image_raw


# %% GET PATHS
to_print = "    Merging TB's TFR data into IET's dataset framework (CBTN dataset into TFRecord dataset (image + age + gradCAM))   "

print(f'\n{"-"*len(to_print)}')
print(to_print)
print(f'{"-"*len(to_print)}\n')

su_debug_flag = True

# --------------------------------------
# read the input arguments and set the base folder
# --------------------------------------
if not su_debug_flag:
    parser = argparse.ArgumentParser(
        description="Merging TB's TFR data into IET's dataset framework (CBTN dataset into TFRecord dataset (image + age + gradCAM))"
    )
    parser.add_argument(
        "-pti",
        "--PATH_TO_IMG_TFR",
        required=True,
        help="Pathe to the folder containing the TFR files of the CBTN dataset.",
    )
    parser.add_argument(
        "-ptg",
        "--PATH_TO_CBTN_GRADCAMS",
        required=False,
        default=None,
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
        "PATH_TO_IMG_TFR": "/flush/iulta54/Research/Data/CBTN/EXTRACTED_SLICES_TFR_TB_20230309/T2",
        "PATH_TO_CBTN_GRADCAMS": None,
        "PATH_TO_TABULAR_DATA": "/flush/iulta54/Research/Data/CBTN/CSV_files/t2_t1c_all_files.xlsx",
        "SAVE_PATH": "/run/media/iulta54/GROUP_HD1/Datasets/CBTN/TFR_DATASET/T2",
    }

# make save path
Path(args_dict["SAVE_PATH"]).mkdir(parents=True, exist_ok=True)

# print input variables
max_len = max([len(key) for key in args_dict])
[print(f"{key:{max_len}s}: {value}") for key, value in args_dict.items()]

# %%  GET FILES TO CONVERT
tfr_img_files = glob.glob(os.path.join(args_dict["PATH_TO_IMG_TFR"], "*.tfrecords"))
if args_dict["PATH_TO_CBTN_GRADCAMS"]:
    gcam_files = glob.glob(os.path.join(args_dict["PATH_TO_CBTN_GRADCAMS"], "*.npy"))

    # check which files have a gradCAM file
    missing_gradCAM = []
    files_to_work_on = []
    for img_f in tfr_img_files:
        # build gradCAM file name
        aus_f = os.path.join(
            args_dict["PATH_TO_CBTN_GRADCAMS"],
            Path(os.path.basename(img_f)).stem + ".npy",
        )
        if aus_f not in gcam_files:
            missing_gradCAM.append(Path(os.path.basename(img_f)).stem)
        else:
            files_to_work_on.append((img_f, aus_f))
else:
    files_to_work_on = [(f, None) for f in tfr_img_files]

print(f"Total number of image files: {len(files_to_work_on)}")
if args_dict["PATH_TO_CBTN_GRADCAMS"]:
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
saved_idx = 0
# load image data and gradCAM for each slice
for idx, f in enumerate(files_to_work_on[0:10]):
    print(
        f"Working on {idx+1:{len(str(len(files_to_work_on)))}d} of {len(files_to_work_on)} (saved {saved_idx+1}) \r",
        end="",
    )

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
    img_data = get_image_from_TFR_file(f[0])
    if args_dict["PATH_TO_CBTN_GRADCAMS"]:
        # load gradCAM
        gradCAM_data = np.load(f[1], allow_pickle=True).item()
        gradCAM_data = [
            np.squeeze(gradCAM_data["gradCAM_raw"][m][:, :, 1, 0])
            for m in range(len(gradCAM_data["gradCAM_raw"]))
        ]
        gradCAM_data = np.mean(
            np.stack(gradCAM_data, axis=-1), axis=-1, dtype="float32"
        )
    else:
        gradCAM_data = np.zeros(img_data.shape, dtype="float32")

    # get age from file
    age_data = int(os.path.basename(f[0]).split("_")[2][0:-1])

    # get ready to write the tf_example into the TFrecord file
    # add infra supra information in the file name
    # TUMOR-TYPE_LOCATION_sID_DAYS_b_brain_MODALITY_rpl_RPL_label_LABEL.tfrecords
    file_name = Path(os.path.basename(f[0])).stem
    if any([label_5_classes == i for i in [1, 3, 5]]):
        file_name = (
            file_name.split("_")[0]
            + "_infra_"
            + "_".join(file_name.split("_")[1:5])
            + f"_{file_name.split('_')[5].upper()}"
            + f"_rlp_{file_name.split('_')[-1]}"
            + "_label_1"
            + ".tfrecords"
        )
        loc = "infra"
    else:
        file_name = (
            file_name.split("_")[0]
            + "_supra_"
            + "_".join(file_name.split("_")[1:5])
            + f"_{file_name.split('_')[5].upper()}"
            + f"_rlp_{file_name.split('_')[-1]}"
            + "_label_1"
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
        "slice_with_tumor": _int64_feature(int(1)),
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
