import os
import random

# import torch
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import glob
from pathlib import Path
import matplotlib.pyplot as plt

from time import time

# %% @ToDo
"""
Build datagenerator object which can be used to get data from TFR or image files
"""


def img_data_generator(
    file_paths,
    normalize_img: bool = True,
    return_gradCAM: bool = False,
    path_to_GradCAMs: str = None,
    return_age: bool = False,
    normalize_age: bool = False,
    batch_size: int = 32,
    buffer_size: int = None,
    nbr_classes: int = 2,
    dataset_type: str = "training",
    rnd_seed: int = 29122009,
    input_size: tuple = (224, 224),
    output_as_RGB: bool = False,
):
    """
    Script that takes the filenames and uses the tf flow_from_dataframe
    to generate a dataset that digests the data.
    @ToDo: make the function pretty.


    Steps
    1 - gets the label for each iamge
    2 - builds the dataframe
    3 - creates datagenerator and applies transformations
    """

    # get labels from file names (this depends on the application)
    # @ToDo: give labels as a list
    start_pattern = "label_"
    end_pattern = "."

    # get the labels
    labels = []
    for f in file_paths:
        start = os.path.basename(f).index(start_pattern)
        stop = os.path.basename(f).rindex(end_pattern)
        labels.append(os.path.basename(f)[start + len(start_pattern) : stop])

    # get the age information
    if return_age:
        # get the age from the file names
        age = [int(os.path.basename(f).split("_")[2][0:-1]) for f in file_paths]
        if normalize_age:
            age = (age - np.mean(age)) / np.std(age)
    else:
        age = [None] * len(labels)

    # get the gradCAM files
    if return_gradCAM:
        # build file names of GradCAM files based on the image files (this makes sure we have all the GradCAMs)
        if path_to_GradCAMs is None:
            raise ValueError(
                f"Datagenerator set to return GradCAM location, but not path to the grad cam files was given!"
            )
        path_to_GradCAM_files = [
            os.path.join(path_to_GradCAMs, f"gradCAM_{os.path.basename(f)}")
            for f in file_paths
        ]
    else:
        path_to_GradCAM_files = [None] * len(labels)

    # build dataframe
    dataset_dataframe = pd.DataFrame(
        {
            "path_to_sample": file_paths,
            "path_to_GradCAM_files": path_to_GradCAM_files,
            "age": age,
            "label": labels,
        }
    )

    # build datagenerator depending on the configurations
    # if dataset_type == "training":
    #     preprocessing = keras.preprocessing.image.ImageDataGenerator(
    #         rescale=1.0 / 255,
    #         shear_range=0.2,
    #         zoom_range=0.2,
    #         rotation_range=45,
    #         horizontal_flip=True,
    #         vertical_flip=True,
    #     )
    # elif all(
    #     [any([dataset_type == "validation", dataset_type == "testing"]), normalize_img]
    # ):
    #     preprocessing = keras.preprocessing.image.ImageDataGenerator(
    #         rescale=1.0 / 255,
    #     )
    # elif all(
    #     [
    #         any([dataset_type == "validation", dataset_type == "testing"]),
    #         not normalize_img,
    #     ]
    # ):
    #     preprocessing = keras.preprocessing.image.ImageDataGenerator(
    #         rescale=1.0,
    #     )
    # else:
    # raise ValueError(
    #     f"Unknown dataset type. Accepted one of the following: training, validation, testing.\nGiven {dataset_type}"
    # )
    preprocessing = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0,
    )

    # build sample generator
    sample_data_generator = preprocessing.flow_from_dataframe(
        dataframe=dataset_dataframe,
        directory=None,
        x_col="path_to_sample",
        y_col="label",
        subset=None,
        batch_size=batch_size,
        seed=rnd_seed if rnd_seed else None,
        shuffle=True if dataset_type == "training" else False,
        class_mode="categorical",
        target_size=input_size,
        color_mode="grayscale" if not output_as_RGB else "rgb",
    )
    return sample_data_generator, len(file_paths) // batch_size


# %% DATA UTILITIES FOR TFR_records

"""
 ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ TFRecord dataset functions

There are two main functions here:
1 - _parse_function -> opens the TFRecord files using the format used during the
                      creation of the TFRecord files. Whithin this function one
                      can manuputale the data in the record to prepare the images
                      and the labels used by the model e.g. add extra channels.
2 - create_dataset -> this looks at all the TFRecords files specified in the dataset
                      and retrievs, shuffles, buffers the data for the model.
                      One here can even implement augmentation if needed. This
                      returns a dataset that the model will use (image, label) format.
The hiper-parameters needed for the preparation of the dataset are:
- batch_size: how many samples at the time should be fed into the model.
- number of parallel loaders: how many files are read at the same time.
- buffer_size: number of samples (individual images) that will be used for the
              shuffling procedure (shuffling is very important!)
"""
import tensorflow as tf
import pandas as pd


def _parse_function_withouth_TF_op(
    proto,
    input_size,
    return_img: bool = True,
    return_gradCAM: bool = False,
    return_age: bool = False,
    nbr_classes: int = 3,
    to_categorical: bool = True,
    output_as_RGB: bool = False,
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
        Tensor of the image encoded in the TFRecord example
    label : tf eager tensor (has the numpy() attribute)
        Tensor of the label encoded in the TFRecord example
    """

    # REFINED VERSION
    key_features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "gradCAM": tf.io.FixedLenFeature([], tf.string),
        "age": tf.io.FixedLenFeature([], tf.int64),
        "file_name": tf.io.FixedLenFeature([], tf.string),
        "slice_with_tumor": tf.io.FixedLenFeature([], tf.int64),
        "label_3_classes": tf.io.FixedLenFeature([], tf.int64),
        "label_5_classes": tf.io.FixedLenFeature([], tf.int64),
    }

    # # TO MATCH TB's DATASET STRUCTURE
    # key_features = {
    #     "image_data": tf.io.FixedLenFeature([], tf.string),
    #     # "gradCAM": tf.io.FixedLenFeature([], tf.string),
    #     "age_days": tf.io.FixedLenFeature([], tf.int64),
    #     # "file_name": tf.io.FixedLenFeature([], tf.string),
    #     # "slice_with_tumor": tf.io.FixedLenFeature([], tf.int64),
    #     "image_label_3_classes": tf.io.FixedLenFeature([], tf.int64),
    #     "image_label_5_classes": tf.io.FixedLenFeature([], tf.int64),
    # }

    # take out specified features (key_features) from the example
    parsed_features = tf.io.parse_single_example(proto, key_features)

    if return_img:
        # decode image from the example (convert from string of bytes and reshape)
        # REFINED VERSION
        image = tf.io.decode_raw(parsed_features["image"], out_type=tf.float32)

        # # TO MATCH TB's DATASET STRUCTURE
        # image = tf.io.decode_raw(parsed_features["image_data"], out_type=tf.float32)

        image = tf.reshape(image, shape=input_size)

    if return_gradCAM:
        # decode image from the example (convert from string of bytes and reshape)
        gradCAM = tf.io.decode_raw(parsed_features["gradCAM"], out_type=tf.float32)
        gradCAM = tf.reshape(gradCAM, shape=input_size)
        gradCAM = tf.cast(gradCAM, dtype=tf.float32)

    if return_age:
        # parsing age
        age = parsed_features["age"]
        age = tf.cast(age, dtype=tf.float64)

    # take out the labels
    if any([nbr_classes == 3, nbr_classes == 5]):
        # REFINED VERSION
        label = (
            tf.cast(parsed_features[f"label_{nbr_classes}_classes"], dtype=tf.int32) - 1
        )

        # # TO MATCH TB's DATASET STRUCTURE
        # label = (
        #     tf.cast(
        #         parsed_features[f"image_label_{nbr_classes}_classes"], dtype=tf.int32
        #     )
        #     - 1
        # )

    else:
        label = tf.cast(parsed_features[f"slice_with_tumor"], dtype=tf.int32)

    if to_categorical:
        label = tf.cast(tf.one_hot(label, nbr_classes), dtype=tf.float32)

    """
    This can get messy. In the end we want that the image is a [H, W, ch], where ch=1 if only 
    the image is returned, ch=2 if image and gradcam is returned and ch=3 if output_as_RGB is True.
    """
    if all([return_img, return_gradCAM, return_age]):
        # fix number of output channels (usually 3 for pre-trained models on imageNet)
        if output_as_RGB:
            image = tf.stack([image, image, gradCAM], axis=-1)
        else:
            image = tf.stack([image, gradCAM], axis=-1)
        return {"image": image, "age": age}, {"label": label}
    elif all([return_img, return_gradCAM, not return_age]):
        if output_as_RGB:
            image = tf.stack([image, image, gradCAM], axis=-1)
        else:
            image = tf.stack([image, gradCAM], axis=-1)
        return {"image": image}, {"label": label}
    elif all([return_img, not return_gradCAM, return_age]):
        if output_as_RGB:
            image = tf.stack([image, image, image], axis=-1)
        else:
            image = tf.expand_dims(image, axis=-1)
        return {"image": image, "age": age}, {"label": label}
    elif all([return_img, not return_gradCAM, not return_age]):
        if output_as_RGB:
            image = tf.stack([image, image, image], axis=-1)
        else:
            image = tf.expand_dims(image, axis=-1)
        return {"image": image}, {"label": label}
    elif all([not return_img, return_gradCAM, return_age]):
        if output_as_RGB:
            gradCAM = tf.stack([gradCAM, gradCAM, gradCAM], axis=-1)
        else:
            gradCAM = tf.expand_dims(gradCAM, axis=-1)
        return ({"image": gradCAM, "age": age}, {"label": label})
    elif all([not return_img, not return_gradCAM, return_age]):
        return {"age": age}, {"label": label}
    else:
        raise ValueError(
            f"Generator set to output NOTHING! Check that this is the intended behviour. If so, continue implementation"
        )


"""
DATA AUGMENTATION
Tha augmentation works as the following
1 - Define the augmentation function that works on the outputs of the _parse_function function e.g. image, label
    Here one can implement all sorts of augmentation using the tf.image functions
2 - On the parsed dataset, use the .map function, where the augmentation function is given. Set the nbr of
    parallel processes.
"""


def to_categorical(sample, label, nbr_classes: int = 3):
    return sample, tf.keras.utils.to_categorical(
        label - tf.constant(1, dtype=tf.int64), nbr_classes
    )


def tfrs_data_generator(
    file_paths,
    input_size,
    dataset_type,
    batch_size,
    buffer_size=5,
    return_img: bool = True,
    return_gradCAM: bool = False,
    return_age: bool = False,
    nbr_classes: int = 3,
    to_categorical: bool = True,
    output_as_RGB: bool = False,
):
    """
    Function that given a list of TFRecord files returns a data generator on them.

    INPUT
    file_paths : list
        List of patsh pointing to the TFRecord files to be included in the generator
    input_size : tuple or list (width, hight)
        Size of the raw image encoded in the TFRecord example (needed to reshape the parsed data)
    dataset_type : string (train, test)
        Specify if the dataset is a training or testing dataset.
        The training dataset is built to iterate infinitely (so that can be used on multiple
        epochs without redifining it) and is shuffled.
    batch_size : int
        Size of each batch.
    buffer_size : int
        Number of samples to use for shuffling. This is very important since allows each batch to
        be as diverse as possible. Note that helps shuffling the input file_paths prior giving
        it to the generator.
    data_augmentation : boolean
        True if data augmentation is to be applied, false if not. Commonly True for training dataset,
        false testing. By default data augmentation if False for test dataset.

    OUTPUT
    dataset : tf.dataset
        A tensorflow dataset which outputs batches of (image, label).

    IMPORTANT
    This generator is quite flexible.
    For example, the labels of the images can be extracted from the file_paths and change
    based on the needs (going from a 3 class classsification to a 5 class classification).
    One can do that  if the information is in the file name. Example of code below:

    dataset = tf.data.TFRecordDataset(file_paths)
    label_from_file_name = [(some_heuristic_on_the_file_names) for file in file_paths]
    dataset = dataset.map(lambda x: (_parse_function(x, image_size=image_size), label_from_file_name),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    NB! The _parse_function should be changed to not output the label.

    []
    """
    # point to the files of the dataset
    dataset = tf.data.TFRecordDataset(file_paths)

    # parce files using the parcer above

    dataset = dataset.map(
        tf.autograph.experimental.do_not_convert(
            lambda x: (
                _parse_function_withouth_TF_op(
                    x,
                    input_size,
                    return_img=return_img,
                    return_gradCAM=return_gradCAM,
                    return_age=return_age,
                    nbr_classes=nbr_classes,
                    to_categorical=to_categorical,
                    output_as_RGB=output_as_RGB,
                )
            )
        ),
        num_parallel_calls=8,
    )

    # make the training dataset to iterate forever
    if dataset_type == "train":
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.repeat()

    # This is gust to make the sample weight to work. keep the input features as dict, but labels take out of the dictionary
    if return_age:
        dataset = dataset.map(
            tf.autograph.experimental.do_not_convert(
                lambda features, targets: (
                    {"image": features["image"], "age": features["age"]},
                    targets["label"],
                )
            )
        )
    else:
        dataset = dataset.map(
            tf.autograph.experimental.do_not_convert(
                lambda features, targets: (
                    {"image": features["image"]},
                    targets["label"],
                )
            )
        )

    # set bach_size
    dataset = dataset.batch(batch_size=batch_size)

    # set how many batches to get ready at the time
    if dataset_type == "train":
        dataset = dataset.prefetch(buffer_size)

    # return the number of steps for this dataset
    # dataset_steps = int(np.ceil(len(file_paths) / batch_size))
    dataset_steps = len(file_paths) // batch_size

    return dataset, dataset_steps


# %% GET FILE NAMES


def get_img_file_names(
    img_dataset_path: str,
    dataset_type: str,
    file_format: str = "jpeg",
    return_labels: bool = False,
    **kwargs,
):
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
    per_class_samples : list
        Returns the list of elements for every class if return_class_elements is True
    """

    # check the optional arguments and initialize them if not present
    # optional arguments
    default_opt_args = {
        "modalities": [
            os.path.basename(f) for f in glob.glob(os.path.join(img_dataset_path, "*"))
        ],
        "task": "detection",
        "nbr_classes": 2,
        "tumor_min_rpl": 0,
        "tumor_max_rpl": 100,
        "brain_min_rpl": 1,
        "brain_max_rpl": 25,
        "tumor_loc": ["infra"],
    }

    if not kwargs:
        kwargs = default_opt_args
    else:
        for keys, values in default_opt_args.items():
            if keys not in kwargs.keys():
                kwargs[keys] = values

    # print(kwargs)

    # initiate variables to be returned
    all_file_names = []

    # work on each dataset type
    if dataset_type.upper() == "BRATS":
        # for modality in kwargs["modalities"]:
        #     # specify if to include slices without tumor
        #     if kwargs["task"].lower() == "detection":
        #         classes = ["slices_without_tumor_label_0", "slices_with_tumor_label_1"]
        #     elif kwargs["task"].lower() == "classification":
        #         classes = ["slices_with_tumor_label_1"]

        #     for s in classes:
        #         files = glob.glob(
        #             os.path.join(img_dataset_path, modality, s, f"*.{file_format}")
        #         )
        #         if s == "slices_with_tumor_label_1":
        #             # filter files with relative position of the tumor
        #             files = [
        #                 (f, int(os.path.basename(f).split("_")[2]))
        #                 for f in files
        #                 if all(
        #                     [
        #                         float(os.path.basename(f).split("_")[5])
        #                         >= kwargs["tumor_min_rpl"],
        #                         float(os.path.basename(f).split("_")[5])
        #                         <= kwargs["tumor_max_rpl"],
        #                     ]
        #                 )
        #             ]
        #         if s == "slices_without_tumor_label_0":
        #             # filter files with relative position of the tumor within [20, 80]
        #             files = [
        #                 (f, int(os.path.basename(f).split("_")[2]))
        #                 for f in files
        #                 if all(
        #                     [
        #                         float(os.path.basename(f).split("_")[5])
        #                         >= kwargs["brain_min_rpl"],
        #                         float(os.path.basename(f).split("_")[5])
        #                         <= kwargs["brain_max_rpl"],
        #                     ]
        #                 )
        #             ]
        #         all_file_names.extend(files)

        # BraTS 2021 has all the files in one folder
        # filter based on the modality (lower_case)
        for modality in kwargs["modalities"]:
            files = glob.glob(os.path.join(img_dataset_path, f"*.{file_format}"))
            files = [f for f in files if modality.lower() in os.path.basename(f)]
            # filter based on task
            if kwargs["task"].lower() == "classification":
                classes = ["label_1"]
            if kwargs["task"].lower() == "detection":
                classes = ["label_0", "label_1"]
            for c in classes:
                # filter based on the class
                aus_files = [f for f in files if c in os.path.basename(f)]
                print(aus_files[0])
                # filter based on the relative position
                if c == "label_1":
                    aus_files = [
                        (f, int(os.path.basename(f).split("_")[1]))
                        for f in aus_files
                        if all(
                            [
                                float(os.path.basename(f).split("_")[4])
                                >= kwargs["tumor_min_rpl"],
                                float(os.path.basename(f).split("_")[4])
                                <= kwargs["tumor_max_rpl"],
                            ]
                        )
                    ]
                elif c == "label_0":
                    aus_files = [
                        (f, int(os.path.basename(f).split("_")[1]))
                        for f in aus_files
                        if all(
                            [
                                float(os.path.basename(f).split("_")[4])
                                >= kwargs["brain_min_rpl"],
                                float(os.path.basename(f).split("_")[4])
                                <= kwargs["brain_max_rpl"],
                            ]
                        )
                    ]
                all_file_names.extend(aus_files)

    if dataset_type.upper() == "CBTN":
        for modality in kwargs["modalities"]:
            # work on the images with tumor

            # REFINED DATASET
            files = glob.glob(
                os.path.join(img_dataset_path, modality, f"*.{file_format}")
            )

            # # THIS IS FOR TAMARAS DATASET
            # files = glob.glob(
            #     os.path.join(img_dataset_path, modality, f"*.{file_format}")
            # )

            # REFINED DATASET
            class_index = 0
            subj_index = 2
            rlp_index = -3
            location_index = 1

            # # TO MATCH TB's DATASET STRUCTURE
            # print(files[0])
            # class_index = 0
            # subj_index = 1
            # rlp_index = -1

            # filter files with relative position of the tumor
            files = [
                (f, os.path.basename(f).split("_")[subj_index])
                for f in files
                if all(
                    [
                        float(Path(os.path.basename(f)).stem.split("_")[rlp_index])
                        >= kwargs["tumor_min_rpl"],
                        float(Path(os.path.basename(f)).stem.split("_")[rlp_index])
                        <= kwargs["tumor_max_rpl"],
                    ]
                )
            ]
            all_file_names.extend(files)

            # # filter based on tumor location
            # # find indexes elements
            # to_keep = []
            # to_remove = []
            # for idx, f in enumerate(all_file_names):
            #     f_loc = os.path.basename(f[0]).split("_")[location_index]
            #     if any([f_loc == l for l in kwargs["tumor_loc"]]):
            #         # print(f"keeping {os.path.basename(f[0])}")
            #         to_keep.append(idx)
            #     else:
            #         # print(f"removing {os.path.basename(f[0])}")
            #         to_remove.append(idx)

            # # print(f"Keeping {len(to_keep)}, removing {len(to_remove)}")
            # all_file_names = [all_file_names[idx] for idx in to_keep]
            # # print(f"{len(all_file_names)} after filtering")

            # add the files without tumor if detection
            if kwargs["task"].lower() == "detection":
                files = glob.glob(
                    os.path.join(img_dataset_path, modality, f"*label_0.{file_format}")
                )
                files = [
                    (f, os.path.basename(f).split("_")[subj_index])
                    for f in files
                    if all(
                        [
                            float(os.path.basename(f).split("_")[rlp_index])
                            >= kwargs["brain_min_rpl"],
                            float(os.path.basename(f).split("_")[rlp_index])
                            <= kwargs["brain_max_rpl"],
                        ]
                    )
                ]
                all_file_names.extend(files)

    if return_labels:
        if kwargs["task"] == "detection":
            labels = [
                1 if "label_1" in os.path.basename(f[0]) else 0 for f in all_file_names
            ]
        elif all(
            [
                kwargs["task"] == "classification",
                kwargs["nbr_classes"] == 3,
            ]
        ):
            labels_3_classes = {
                "ASTROCYTOMA": 0,
                "EPENDYMOMA": 1,
                "MEDULLOBLASTOMA": 2,
            }
            labels = [
                labels_3_classes[os.path.basename(f[0]).split("_")[class_index]]
                for f in all_file_names
            ]

        elif all(
            [
                kwargs["task"] == "classification",
                kwargs["nbr_classes"] == 5,
            ]
        ):
            labels_5_classes = {
                "ASTROCYTOMA_infra": 0,
                "ASTROCYTOMA_supra": 1,
                "EPENDYMOMA_infra": 2,
                "EPENDYMOMA_supra": 3,
                "MEDULLOBLASTOMA_infra": 4,
            }
            labels = [
                labels_5_classes["_".join(os.path.basename(f[0]).split("_")[0:2])]
                for f in all_file_names
            ]

        return [(f[0], f[1], l) for f, l in zip(all_file_names, labels)]
    else:
        return all_file_names


# %% GET STATISTICS FOR DATA NORMALIZATION


def get_normalization_values(
    dataset,
    dataset_steps,
    return_gradCAM_norm_values: bool = False,
    return_age_norm_values: bool = False,
):
    """
    Stript that loops trough the dataset and gets the mean and std used for normalizing the
    images, gradcams and age if required
    """

    img, gradCAM, age = [], [], []
    ds_iter = iter(dataset)
    for _ in range(dataset_steps):
        sample = next(ds_iter)
        img.append(sample[0]["image"][:, :, :, 0])
        if return_gradCAM_norm_values:
            gradCAM.append(sample[0]["image"][:, :, :, 1])
        if return_age_norm_values:
            age.append(sample[0]["age"])

    img_mean = np.mean(np.vstack(img))
    img_std = np.std(np.vstack(img))
    img_stats = (img_mean, img_std)

    if return_gradCAM_norm_values:
        gradCAM_mean = np.mean(np.vstack(gradCAM))
        gradCAM_std = np.std(np.vstack(gradCAM))
        gradCAM_stats = (gradCAM_mean, gradCAM_std)

    if return_age_norm_values:
        age_mean = np.mean(np.vstack(age))
        age_std = np.std(np.vstack(age))
        age_stats = (age_mean, age_std)

    if all([return_gradCAM_norm_values, not return_age_norm_values]):
        return img_stats, gradCAM_stats, None
    elif all([not return_gradCAM_norm_values, return_age_norm_values]):
        return img_stats, None, age_stats
    elif all([return_gradCAM_norm_values, return_age_norm_values]):
        return img_stats, gradCAM_stats, age_stats
    else:
        return img_stats, None, None


def plot_tfr_dataset_intensity_dist(
    tf_dataset,
    tf_dataset_steps,
    save_path: str = None,
    plot_name: str = "Dataset_intensity_distribution",
    background_value: float = 0.0,
):
    """
    Plots the dataset intensity distribution given a tf_dataset that can be iterated
    """
    samples = []
    ds_iter = iter(tf_dataset)

    for i in range(tf_dataset_steps):
        x, y = next(ds_iter)
        samples.append(x["image"].numpy())
    samples = np.vstack(samples)

    # plot histogram
    fig = plt.figure()
    plt.hist(
        np.ma.masked_where(samples.ravel() == background_value, samples.ravel()),
        bins=256,
    )
    plt.title(plot_name)
    if save_path:
        fig.savefig(os.path.join(save_path, f"{plot_name}_distribution.png"))
        plt.close(fig)
    else:
        plt.show()


# %% PLOTTING UTILITIES


def show_batched_example_tfrs_dataset(
    dataset,
    recipe,
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
        fig, ax = plt.subplots(nrows=1, ncols=2 if show_gradCAM else 1, figsize=(5, 5))

        if show_gradCAM:
            ax[0].imshow(image_batch["image"][i, :, :, 0], cmap="gray")
            label = label_batch[i]
            ax[0].set_title(f"{label}, {class_names[label.argmax()]}")
            ax[1].imshow(image_batch["image"][i, :, :, 1], cmap="gray")
        else:
            ax.imshow(image_batch["image"][i, :, :, 0], cmap="gray")
            label = label_batch[i]
            ax.set_title(f"{label}, {class_names[label.numpy().argmax()]}")
        fig.savefig(os.path.join(recipe["SAVE_PATH"], f"example_{i+1:4d}.png"))
        plt.close(fig)

    if show_histogram:
        fig, ax = plt.subplots(nrows=1, ncols=2 if show_gradCAM else 1, figsize=(5, 5))
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
            fig.savefig(os.path.join(recipe["SAVE_PATH"], "histogram.png"))
            plt.close(fig)


# %% DATA UTILITIES FOR THE MULTIPLE INSTANCE LEARNING TRAINING
def get_subject_bag_enc(
    subject_file_paths,
    enc_model,
    config_file,
    bag_size: int = 5,
    sort_by_location: bool = False,
    shuffle_samples_in_bag: bool = True,
    rnd_seed: int = None,
    imageNet_pretrained_encoder: bool = False,
    augment_instances: bool = False,
):
    """
    Given a list of files, the encoding model and the configuration file which was used to train the
    encoder model, returns a numpy array with size [bag_size, enc_dim] containing the encodings of the subject images.

    Steps:
    - create generator to consume the subject files
    - encode the images
    - aggregate in one bag
    """
    if sort_by_location:
        # sort file names based on the location (from the center of the tumor)
        slice_position_index = -3
        slice_position = [
            int(Path(os.path.basename(f)).stem.split("_")[slice_position_index])
            for f in subject_file_paths
        ]
        # adapt to have the slices with relative position ~50 to be the first ones
        slice_position = np.abs(np.array(slice_position) - 50)
        sorted_slices = np.argsort(slice_position)
        subject_file_paths = [subject_file_paths[i] for i in sorted_slices]

    # fix random seed
    if rnd_seed:
        np.random.seed(rnd_seed)

    # fix the bag size (work on the files before creating the generator and predicting)
    if len(subject_file_paths) < bag_size:
        for i in range(bag_size - len(subject_file_paths)):
            if sort_by_location:
                # replicate more of the central (first) slices
                slice_index = i
            else:
                # randomly oversample to get the right number of instantces for the bag
                slice_index = np.random.randint(len(subject_file_paths))
            subject_file_paths.append(subject_file_paths[slice_index])
    elif len(subject_file_paths) > bag_size:
        aus_subject_file_paths = []
        if sort_by_location:
            # replicate more of the central (first) slices
            slice_index = range(bag_size)
        else:
            # randomly oversample to get the right number of instantces for the bag
            slice_index = np.random.randint(
                enc_images.shape[0],
                size=bag_size,
            )
        for idx in slice_index:
            aus_subject_file_paths.append(subject_file_paths[idx])

        subject_file_paths = aus_subject_file_paths

    if shuffle_samples_in_bag:
        random.shuffle(subject_file_paths)

    # build generator
    target_size = (224, 224)
    img_gen, _ = tfrs_data_generator(
        file_paths=subject_file_paths,
        input_size=target_size,
        batch_size=len(subject_file_paths),
        buffer_size=10,
        return_gradCAM=config_file["USE_GRADCAM"],
        return_age=config_file["USE_AGE"],
        dataset_type="test",
        nbr_classes=config_file["NBR_CLASSES"],
        output_as_RGB=imageNet_pretrained_encoder,
    )

    # get out samples from the generator
    sample = next(iter(img_gen))
    bag_imgs, bag_label = sample[0]["image"].numpy(), sample[1][0].numpy()

    # for each image get the encoded version using the encoding model
    if augment_instances:
        aug_instances = tf.image.random_flip_left_right(
            sample[0]["image"], seed=rnd_seed
        )
        aug_instances = tf.image.random_flip_up_down(aug_instances, seed=rnd_seed)
        enc_images = enc_model.predict(aug_instances, verbose=0)
    else:
        enc_images = enc_model.predict(sample[0], verbose=0)
    return (enc_images, bag_label.squeeze()), bag_imgs.squeeze()


def get_train_data_MIL_model(
    per_subject_files_dict: dict,
    sample_encoder_model,
    data_gen_configuration_dict: dict,
    bag_size: int = 10,
    sort_by_slice_location: bool = True,
    shuffle_samples_in_bag: bool = True,
    debug_number_of_bags: int = None,
    rnd_seed: int = None,
    imageNet_pretrained_encoder: bool = True,
    augment_instances: bool = False,
):
    """
    Utility that given a dictionary with the list of files for each subject, creates a dataset ready for training
    where each sample in the dataset is a bag of encoded samples from each subjects.

    INPUTS
        per_subject_files_dict: (dict) dictionaty where every key is one of the subjects with value the list of files for the subject
        sample_encoder_model: depp learning model to ensoce every sample in the bag
        bag_size: (int) number of samples in each bag
        sort_by_slice_location: (bool) specify if the samples in the bag are the slices most central in the tumor
        shuffle_samples_in_bag: (bool) if the samples in the bag should shuffled.

    OUTPUT
        bags: (list) list of bags each containing the samples for each subject
        bags_label: (list) list of labels for each bag alligned with the bags output
        bags_images: (list) list of bags each containing the images used to obtain the encoded samples
    """

    # TRAIN DATA
    bags, bags_label, bags_images = [], [], []

    for idx, subject_files in enumerate(per_subject_files_dict.values()):
        print(f"Working on subject {idx+1:} of {len(per_subject_files_dict)}\r", end="")
        (bag, labels), images = get_subject_bag_enc(
            subject_files,
            sample_encoder_model,
            data_gen_configuration_dict,
            bag_size=bag_size,
            sort_by_location=sort_by_slice_location,
            shuffle_samples_in_bag=shuffle_samples_in_bag,
            rnd_seed=rnd_seed,
            imageNet_pretrained_encoder=imageNet_pretrained_encoder,
            augment_instances=augment_instances,
        )
        bags.append(bag)
        bags_label.append(labels)
        bags_images.append(images)
        if debug_number_of_bags:
            if (idx + 1) == debug_number_of_bags:
                break
    print("\n")

    # shuffle bags
    if rnd_seed:
        random.seed(rnd_seed)

    zipped = list(zip(bags, bags_label, bags_images))
    random.shuffle(zipped)
    bags, bags_label, bags_images = zip(*zipped)

    # reshape to have the bag dimension as first element and the number of bags as second element (from keras implementation)
    bags = list(np.swapaxes(bags, 0, 1))
    bags_label = np.array(bags_label)

    #

    return bags, bags_label, bags_images
