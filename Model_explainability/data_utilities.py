import os
import random

# import torch
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import glob

from time import time

# %% @ToDo
"""
Build datagenerator object which can be used to get data from TFR or image files
"""


def img_data_generator(
    sample_files,
    use_GradCAM_loc: bool = False,
    path_to_GradCAMs: str = None,
    use_age: bool = False,
    normalize_age: bool = True,
    batch_size: int = 32,
    dataset_type: str = "training",
    rnd_seed: int = 29122009,
    target_size: tuple = (224, 224),
):
    """
    Script that takes the filenames and uses the tf flow_from_dataframe
    to generate a dataset that digests the data.
    @ToDo: pake the function pretty.


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
    for f in sample_files:
        start = os.path.basename(f).index(start_pattern)
        stop = os.path.basename(f).rindex(end_pattern)
        labels.append(os.path.basename(f)[start + len(start_pattern) : stop])

    # get the age information
    if use_age:
        # get the age from the file names
        age = [int(os.path.basename(f).split("_")[2][0:-1]) for f in sample_files]
        if normalize_age:
            age = (age - np.mean(age)) / np.std(age)
    else:
        age = [None] * len(labels)

    # get the gradCAM files
    if use_GradCAM_loc:
        # build file names of GradCAM files based on the image files (this makes sure we have all the GradCAMs)
        if path_to_GradCAMs is None:
            raise ValueError(
                f"Datagenerator set to return GradCAM location, but not path to the grad cam files was given!"
            )
        path_to_GradCAM_files = [
            os.path.join(path_to_GradCAMs, f"gradCAM_{os.path.basename(f)}")
            for f in sample_files
        ]
    else:
        path_to_GradCAM_files = [None] * len(labels)

    # build dataframe
    dataset_dataframe = pd.DataFrame(
        {
            "path_to_sample": sample_files,
            "path_to_GradCAM_files": path_to_GradCAM_files,
            "age": age,
            "label": labels,
        }
    )

    # build datagenerator depending on the configurations
    if dataset_type == "training":
        preprocessing = keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=45,
            horizontal_flip=True,
            vertical_flip=True,
        )
    elif any([dataset_type == "validation", dataset_type == "testing"]):
        preprocessing = keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
        )

    else:
        raise ValueError(
            f"Unknown dataset type. Accepted one of the following: training, validation, testing.\nGiven {dataset_type}"
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
        target_size=target_size,
        color_mode="grayscale",
    )

    #     # build age generator
    #     if use_age:
    #         age_data_generator = tf.data.Dataset.from_tensor_slices(age)
    #         age_data_generator = age_data_generator.batch(batch_size).shuffle(
    #     buffer_size, seed=None, reshuffle_each_iteration=None, name=None
    # )

    #     # build generator for GradCAM
    #     if use_GradCAM_loc:
    #         gradCAM_data_generator = preprocessing.flow_from_dataframe(
    #         dataframe=dataset_dataframe,
    #         directory=None,
    #         x_col="path_to_GradCAM_files",
    #         y_col="label",
    #         subset=None,
    #         batch_size=batch_size,
    #         seed=rnd_seed if rnd_seed else None,
    #         shuffle=True if dataset_type == "training" else False,
    #         class_mode="categorical",
    #         target_size=target_size,
    #         color_mode="grayscale",
    #         seed = rnd_seed,
    #     )

    # @ToDo compute normalization parameters and apply to the generator.
    return sample_data_generator


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


def _parse_function(
    proto,
    input_size,
    return_img: bool = True,
    normalize_img: bool = True,
    img_norm_values: tuple = None,
    return_gradCAM: bool = False,
    normalize_gradCAM: bool = True,
    gradCAM_norm_values: tuple = None,
    return_age: bool = False,
    normalize_age: bool = False,
    age_norm_values: tuple = None,
    nbr_classes: int = 3,
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

    key_features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "gradCAM": tf.io.FixedLenFeature([], tf.string),
        "age": tf.io.FixedLenFeature([], tf.int64),
        "file_name": tf.io.FixedLenFeature([], tf.string),
        "slice_with_tumor": tf.io.FixedLenFeature([], tf.int64),
        "label_3_classes": tf.io.FixedLenFeature([], tf.int64),
        "label_5_classes": tf.io.FixedLenFeature([], tf.int64),
    }

    # take out specified features (key_features) from the example
    parsed_features = tf.io.parse_single_example(proto, key_features)
    if return_img:
        # decode image from the example (convert from string of bytes and reshape)
        image = tf.io.decode_raw(parsed_features["image"], out_type=tf.float32)
        image = tf.reshape(image, shape=input_size)
        if normalize_img:
            if img_norm_values:
                image = tf.math.divide(
                    tf.math.subtract(image, img_norm_values[0]), img_norm_values[1]
                )
            else:
                image = tf.math.divide(image, 255)
                image = tf.math.multiply(image, 2)
                image = tf.math.subtract(image, 1)
        # image = tf.stack([image, image, image], axis=-1)
        # image = tf.image.resize(image, input_size)

    if return_gradCAM:
        # decode image from the example (convert from string of bytes and reshape)
        gradCAM = tf.io.decode_raw(parsed_features["gradCAM"], out_type=tf.float32)
        gradCAM = tf.reshape(gradCAM, shape=input_size)
        if normalize_gradCAM:
            if gradCAM_norm_values:
                gradCAM = tf.math.divide(
                    tf.math.subtract(gradCAM, gradCAM_norm_values[0]),
                    gradCAM_norm_values[1],
                )
            else:
                gradCAM = tf.math.divide(gradCAM, 255)
                gradCAM = tf.math.multiply(gradCAM, 2)
                gradCAM = tf.math.subtract(gradCAM, 1)
        gradCAM = tf.cast(gradCAM, dtype=tf.float32)
        # gradCAM = tf.image.resize(gradCAM, gradCAM)

    if return_age:
        # parsing age
        age = parsed_features["age"]
        age = tf.cast(age, dtype=tf.float64)
        if normalize_age:
            age = tf.math.divide(
                tf.math.subtract(age, age_norm_values[0]), age_norm_values[1]
            )

    # take out the labels
    if any([nbr_classes == 3, nbr_classes == 5]):
        label = parsed_features[f"label_{nbr_classes}_classes"]
        label = label - 1
    else:
        label = parsed_features[f"slice_with_tumor"]
    # convert to categorical
    label = tf.cast(tf.one_hot(tf.cast(label, tf.int32), nbr_classes), dtype=tf.float32)

    if all([return_img, return_gradCAM, return_age]):
        image = tf.stack([image, gradCAM], axis=-1)
        return {"image": image, "age": age}, {"label": label}
    elif all([return_img, return_gradCAM, not return_age]):
        image = tf.stack([image, gradCAM], axis=-1)
        return {"image": image}, {"label": label}
    elif all([return_img, not return_gradCAM, return_age]):
        return {"image": tf.expand_dims(image, axis=-1), "age": age}, {"label": label}
    elif all([return_img, not return_gradCAM, not return_age]):
        return {"image": tf.expand_dims(image, axis=-1)}, {"label": label}
    elif all([not return_img, return_gradCAM, return_age]):
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


def flip(sample, label, random=0.5):
    # perform random flip with probability random
    if tf.random.uniform(()) <= random:
        aug_img = tf.expand_dims(sample["image"], 0)
        # perform augmentation
        aug_img = tf.image.flip_left_right(aug_img)
        # perform augmentation
        # aug_img = tf.image.flip_up_down(aug_img)
        # sample["image"] = aug_img
        sample["image"] = tf.squeeze(aug_img, axis=0)

        return sample, label
    else:
        return sample, label


def brightness(sample, label, max_delta=0.5, random=0.5):
    # perform random brightness with probability random
    if tf.random.uniform(()) <= random:
        # aug_img = tf.expand_dims(sample["image"], 0)
        aug_img = sample["image"]
        print(aug_img.shape)
        # perform augmentation
        aug_img = tf.image.random_brightness(aug_img, max_delta=max_delta)
        # sample["image"] = tf.squeeze(aug_img)
        sample["image"] = aug_img
        return sample, label
    else:
        return sample, label


def rotation(sample, label, random=0.5):
    # perform random rotation with probability random
    if tf.random.uniform(()) <= random:
        aug_img = tf.expand_dims(sample["image"], 0)
        if tf.random.uniform(()) <= 0.5:
            aug_img = tf.image.rot90(aug_img, k=1)
        else:
            aug_img = tf.image.rot90(aug_img, k=3)
        sample["image"] = tf.squeeze(aug_img, axis=0)
        return sample, label
    else:
        return sample, label


def to_categorical(sample, label, nbr_classes: int = 3):
    return sample, tf.keras.utils.to_categorical(
        label - tf.constant(1, dtype=tf.int64), nbr_classes
    )


"""
Use the above to create the actual data generator
"""


def tfrs_data_generator(
    file_paths,
    input_size,
    dataset_type,
    batch_size,
    buffer_size=5,
    data_augmentation=True,
    return_img: bool = True,
    normalize_img: bool = True,
    img_norm_values: tuple = None,
    normalize_gradCAM: bool = True,
    gradCAM_norm_values: tuple = None,
    return_gradCAM: bool = False,
    return_age: bool = False,
    normalize_age: bool = False,
    age_norm_values: tuple = None,
    nbr_classes: int = 3,
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
                _parse_function(
                    x,
                    input_size,
                    return_img=return_img,
                    normalize_img=normalize_img,
                    img_norm_values=img_norm_values,
                    return_gradCAM=return_gradCAM,
                    normalize_gradCAM=normalize_gradCAM,
                    gradCAM_norm_values=gradCAM_norm_values,
                    return_age=return_age,
                    normalize_age=normalize_age,
                    age_norm_values=age_norm_values,
                    nbr_classes=nbr_classes,
                )
            )
        ),
        num_parallel_calls=8,
    )

    # make the training dataset to iterate forever

    if dataset_type == "train":
        dataset = dataset.repeat()

    # shuffle the training dataset
    if dataset_type != "test":
        dataset = dataset.shuffle(buffer_size=buffer_size)

    """ DATA AUGMENTATION """
    if all([dataset_type != "test", data_augmentation == True, return_img]):
        # implement data augmentation
        # dataset = dataset.map(flip)
        dataset = dataset.map(rotation)
        dataset = dataset.map(flip)
        # dataset = dataset.map(brightness)

    # set bach_size
    dataset = dataset.batch(batch_size=batch_size)

    # set how many batches to get ready at the time
    if dataset_type == "train":
        dataset = dataset.prefetch(buffer_size)
        # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # return the number of steps for this dataset
    dataset_steps = len(file_paths) // batch_size

    return dataset, dataset_steps


# %% GET FILE NAMES


def get_img_file_names(
    img_dataset_path: str,
    dataset_type: str,
    file_format: str = "jpeg",
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
                files = glob.glob(
                    os.path.join(img_dataset_path, modality, s, f"*.{file_format}")
                )
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
            files = glob.glob(
                os.path.join(img_dataset_path, modality, f"*label_1.{file_format}")
            )
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
                    os.path.join(img_dataset_path, modality, f"*label_0.{file_format}")
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
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    next_element = iterator.get_next()
    with tf.compat.v1.Session() as sess:
        img, gradCAM, age = [], [], []
        for _ in range(dataset_steps):
            sample = sess.run(next_element)
            img.append(sample[0]["image"][:, :, :, 0])
            if return_gradCAM_norm_values:
                gradCAM.append(sample[0]["image"][:, :, :, 1])
            if return_age_norm_values:
                age.append(sample[0]["age"])

    img_mean = np.mean(np.vstack(img))
    img_std = np.std(np.vstack(img))
    img_stats = (img_mean, img_std)

    if return_gradCAM_norm_values:
        print(gradCAM[0].mean())
        gradCAM_mean = np.mean(np.vstack(gradCAM))
        gradCAM_std = np.std(np.vstack(gradCAM))
        gradCAM_stats = (gradCAM_mean, gradCAM_std)

    if return_age_norm_values:
        age_mean = np.mean(np.vstack(age))
        age_std = np.std(np.vstack(age))
        age_stats = (age_mean, age_std)

    if all([return_gradCAM_norm_values, not return_age_norm_values]):
        return img_stats, gradCAM_stats
    elif all([not return_gradCAM_norm_values, return_age_norm_values]):
        return img_stats, age_stats
    elif all([return_gradCAM_norm_values, return_age_norm_values]):
        return img_stats, gradCAM_stats, age_stats
    else:
        return img_stats
