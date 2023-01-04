import os
import random

# import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils

from time import time

# from batchgenerators.augmentations.crop_and_pad_augmentations import crop
# from batchgenerators.augmentations.utils import pad_nd_image
# from batchgenerators.dataloading.data_loader import DataLoader
# from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
# from batchgenerators.examples.brats2017.brats2017_dataloader_3D import (
#     get_list_of_patients,
#     BraTS2017DataLoader3D,
#     get_train_transform,
# )
# from batchgenerators.examples.brats2017.config import (
#     brats_preprocessed_folder,
#     num_threads_for_brats_example,
# )
# from batchgenerators.utilities.data_splitting import get_split_deterministic


# class TumorDetectionDataset2D(Dataset):

#     """Dataset of 2D images for tumor detection"""

#     def __init__(
#         self,
#         sample_files: list,
#         label_from_file: bool = False,
#         labels: list = None,
#         nbr_classes: int = 2,
#         labels_to_categorical: bool = False,
#         transform=None,
#     ):
#         """
#         Args:
#             sample_files (list): list of the files to use in this generatoe
#             label_from_file (bool, optional): if the labels are infered from the file name or are given as a list
#             labels (list, optional): if the label_from_file=False, labels are expected to be given for each sample (check allignment)
#             nbr_classes (int, optional): used when labels_from_file is True and labels_to_categorical is True.
#             labels_to_categorical (bool, optional): specify if the labels should be transformed to categorical
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.

#         NOTE
#         This implementation expect the labels to be integers. If this is ot the case, one needs to implement
#         a conversion between the labels (strings or others) into integers of the different classes
#         """
#         self.sample_files = sample_files
#         self.label_from_file = label_from_file

#         if all([not self.label_from_file, labels]):
#             # labels are given, check that samples and labels have the same size
#             assert isinstance(labels, list), "The given labels are not stored in a list"
#             assert len(labels) == len(self.sample_files), (
#                 f"Samle files and labels have different lenghts.\n"
#                 f"Given {len(self.sample_files)} samples and {len(labels)}."
#             )
#             self.labels = labels
#             # infere number of classes from the labels
#             self.nbr_classes = np.unique(self.labels)
#         else:
#             self.labels = None
#             self.nbr_classes = nbr_classes

#         # transformations
#         self.labels_to_categorical = labels_to_categorical
#         self.transform = transform

#     def __len__(self):
#         return len(self.sample_files)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         # load image
#         image = io.imread(self.sample_files[idx])
#         # check if image has [d1, d2, C]. If not, add extra dimentions to [d1, d2, C]
#         if len(image.shape) == 2:
#             image = image[:, :, np.newaxis]

#         # get or load label
#         if self.label_from_file:
#             # heuristig to get the label from the file name
#             start = os.path.basename(self.sample_files[idx]).index("label_")
#             stop = os.path.basename(self.sample_files[idx]).rindex(".")
#             # label = int(os.path.basename(self.sample_files[idx])[start+6:stop])
#             label = torch.tensor(
#                 int(os.path.basename(self.sample_files[idx])[start + 6 : stop]),
#                 dtype=torch.float32,
#             )
#         else:
#             label = torch.tensor(int(self.labels[idx]), dtype=torch.float32)
#         # transform to categorical if neede
#         if self.labels_to_categorical:
#             label = torch.nn.functional.one_hot(
#                 label.long(), num_classes=self.nbr_classes
#             )

#         if self.transform:

#             image = self.transform(image)

#         # create sample
#         sample = {"image": image, "label": label.to(dtype=torch.float32)}

#         return sample


# def batch_mean_and_sd(loader):
#     cnt = 0
#     fst_moment = torch.empty(3)
#     snd_moment = torch.empty(3)

#     for sample in loader:
#         b, _, h, w = sample["image"].shape
#         nb_pixels = b * h * w
#         sum_ = torch.sum(sample["image"], dim=[0, 2, 3])
#         sum_of_square = torch.sum(sample["image"] ** 2, dim=[0, 2, 3])
#         fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
#         snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
#         cnt += nb_pixels

#     mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment**2)
#     return mean, std


# # # %% TRY OUT DATA GENERATOR FROM nnUNet


# # class BraTSTumorDetection2D(DataLoader):
# #     def __init__(
# #         self,
# #         data,
# #         batch_size,
# #         patch_size,
# #         num_threads_in_multithreaded,
# #         seed_for_shuffle=1234,
# #         return_incomplete=False,
# #         shuffle=True,
# #         label_from_file: bool = False,
# #         labels_to_categorical: bool = False,
# #     ):
# #         """
# #         data must be a list data to be loaded (here it can be modified to get the slices from a 3D volume in case)
# #         patch_size is the spatial size the retured batch will have
# #         """
# #         super().__init__(
# #             data,
# #             batch_size,
# #             num_threads_in_multithreaded,
# #             seed_for_shuffle,
# #             return_incomplete,
# #             shuffle,
# #             True,
# #         )
# #         self.patch_size = patch_size
# #         self.num_modalities = 1
# #         self.indices = list(range(len(data)))

# #     @staticmethod
# #     def load_patient(sample):
# #         # here is the heuristic on how an image is loaded
# #         return io.imread(sample)

# #     def generate_train_batch(self):
# #         # DataLoader has its own methods for selecting what patients to use next, see its Documentation
# #         idx = self.get_indices()
# #         sample_for_batch = [self._data[i] for i in idx]

# #         # initialize empty array for data and label
# #         data = np.zeros(
# #             (self.batch_size, self.num_modalities, *self.patch_size), dtype=np.float32
# #         )
# #         label = np.zeros((self.batch_size, 1), dtype=np.float32)

# #         sample_names = []

# #         # iterate over patients_for_batch and include them in the batch
# #         for i, j in enumerate(sample_for_batch):
# #             sample_data = self.load_patient(j)

# #             # this will only pad sample_data if its shape is smaller than self.patch_size
# #             sample_data = pad_nd_image(sample_data, self.patch_size)
# #             # sample now has shape (x,y) but it needs to be (b, c, x, y) so add extra dimensions
# #             sample_data = sample_data[np.newaxis, np.newaxis, :, :]

# #             # now random crop to self.patch_size
# #             # crop expects the data to be (b, c, x, y, z) but patient_data is (c, x, y, z) so we need to add one
# #             # dummy dimension in order for it to work (@Todo, could be improved)
# #             sample_data, _ = crop(
# #                 data=sample_data,
# #                 seg=None,
# #                 crop_size=self.patch_size,
# #                 crop_type="random",
# #             )

# #             data[i] = sample_data[0]
# #             sample_names.append(os.path.basename(j))

# #         return {"data": data, "names": sample_names}


def get_data_generator_TF(
    sample_files,
    batch_size: int = 32,
    dataset_type: str = "training",
    rnd_seed: int = 29122009,
    target_size: tuple = (224, 224),
):
    """
    Script that takes the filenames and uses the tf flow_from_dataframe
    to generate a dataset that digests the data.

    Steps
    1 - gets the label for each iamge
    2 - builds the dataframe
    3 - creates datagenerator and applies transformations
    """

    # get labels from file names (this depends on the application)
    # @ToDo: give labels as a list
    start_pattern = "label_"
    end_pattern = "."

    labels = []
    for f in sample_files:
        start = os.path.basename(f).index(start_pattern)
        stop = os.path.basename(f).rindex(end_pattern)
        labels.append(os.path.basename(f)[start + len(start_pattern) : stop])

    # build dataframe using
    dataset_dataframe = pd.DataFrame(
        {
            "path_to_sample": sample_files,
            "label": labels,
        }
    )

    # build datagenerator
    if dataset_type == "training":
        data_generator = keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            # shear_range=0.2,
            # zoom_range=0.2,
            # rotation_range=45,
            # horizontal_flip=True,
            # vertical_flip=True,
        )

    elif any([dataset_type == "validation", dataset_type == "testing"]):
        data_generator = keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
        )

    else:
        raise ValueError(
            f"Unknown dataset type. Accepted one of the following: training, validation, testing.\nGiven {dataset_type}"
        )

    data_generator = data_generator.flow_from_dataframe(
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

    # @ToDo compute normalization parameters and apply to the generator.
    # If the dataset_type is not Trraining, nroamlization parameters need to be given

    return data_generator
