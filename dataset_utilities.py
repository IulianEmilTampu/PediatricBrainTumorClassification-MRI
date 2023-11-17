import os
import sys
import glob
import PIL
import pathlib
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Any, Tuple, Union

from sklearn.model_selection import (
    KFold,
    train_test_split,
    StratifiedKFold,
    StratifiedGroupKFold,
    GroupShuffleSplit,
    StratifiedShuffleSplit,
)
from sklearn.utils import class_weight

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets
import torchvision.transforms as T

import pytorch_lightning as pl


def _get_split(
    config: dict,
    heuristic_for_file_discovery,
    heuristic_for_subject_ID_extraction,
    heuristic_for_class_extraction,
    heuristic_for_relative_position_extraction=None,
    repetition_number=1,
    randomize_subject_labels: bool = False,
    randomize_slice_labels: bool = False,
    select_slices_in_range: Union[list, tuple, float, int] = None,
):
    """
    Utility that given the path to the dataset, creates testing, training and validation splits.
    This Utility creates a pandas dataframe where the information from every file is first
    stored. Then the split is performed using the information from the dataframe.
    The dataframe has at least two columns: subject_ID and file_path. A third column can be
    added describing the class (if classification is the end task).

    The utility returns a pandas dataframe where, in addition to the previous columns, the set of belonging of each file
    is stored for each cross validation fold (one column for each fold).
    """

    def _get_dataframe_from_dataset_folder(
        path_to_dataset,
        heuristic_for_file_discovery,
        heuristic_for_subject_ID_extraction,
        heuristic_for_class_extraction=None,
        heuristic_for_relative_position_extraction=None,
    ):
        # get all the files in the dataset_folder
        all_files = sorted(
            p
            for p in Path(path_to_dataset).glob(heuristic_for_file_discovery)
            if not p.name.startswith(".")
        )
        # get all the subject_IDs using the heuristic_for_subject_ID_extraction
        subject_IDs = [heuristic_for_subject_ID_extraction(str(f)) for f in all_files]

        if heuristic_for_class_extraction:
            per_file_class = [heuristic_for_class_extraction(str(f)) for f in all_files]
        else:
            per_file_class = None

        # get relative position information
        if heuristic_for_relative_position_extraction:
            per_file_rlp = [
                heuristic_for_relative_position_extraction(str(f)) for f in all_files
            ]
        else:
            per_file_rlp = None

        # build dataframe and return
        return pd.DataFrame(
            {
                "subject_IDs": subject_IDs,
                "file_path": all_files,
                "target": per_file_class,
                "tumor_relative_position": per_file_rlp,
            }
        )

    def _select_slices_in_range(dataset_df, select_slices_in_range):
        """
        Utility that given a range of relative positions with respect to the tumor,
        select only those slices in the range.
        """
        # check if the range is one or two values.
        # If two value set the same value as min and max
        if isinstance(select_slices_in_range, (int, float)):
            select_slices_in_range = (select_slices_in_range, select_slices_in_range)
        elif len(select_slices_in_range) == 1:
            select_slices_in_range = (
                select_slices_in_range[0],
                select_slices_in_range[0],
            )

        # check if the given dataframe has the relative position column
        if not any(dataset_df.tumor_relative_position):
            raise ValueError(
                "The given dataset does not have the tumor_relative_position informaiton for the different samples. CHeck that the _get_dataframe_from_dataset_folder was flagged to extract the relative position information from the fine names."
            )

        # apply filter to the dataframe
        dataset_df = dataset_df.loc[
            (dataset_df["tumor_relative_position"] >= select_slices_in_range[0])
            & (dataset_df["tumor_relative_position"] <= select_slices_in_range[1])
        ]

        return dataset_df

    def _randomize_labels(
        dataset_df,
        randomize_subject_labels: bool = False,
        randomize_slice_labels: bool = False,
        seed: int = 20230429,
    ):
        """
        Utility that randomizes the classes subject-wise or slice wise.
        """
        # fix random number
        np.random.seed(seed)

        # check that the given dataframe has the target information
        if not any(dataset_df.target):
            raise ValueError(
                "The given dataframe does not have the infromation regarding the target of each file. MAke sure that this information was extracted using the _get_dataframe_from_dataset_folder function."
            )

        # do per slice normalization if requested (skip the per subject)
        if randomize_slice_labels:
            aus_targets = dataset_df["target"].values
            # shuffle
            np.random.shuffle(aus_targets)
            # and put back
            dataset_df["target"] = pd.Series(aus_targets).values
        elif randomize_subject_labels:
            # shuffle subject-wise (the slices for each subject have the same random tanger).
            # Function to randomize targets within each group
            def randomize_targets(group, target_to_chose_from):
                # select random target for this group
                random_target = np.random.choice(target_to_chose_from, size=1)
                random_target = pd.Series(random_target).repeat(len(group)).values
                group["target"] = random_target
                return group

            dataset_df = dataset_df.groupby("subject_IDs", group_keys=False).apply(
                randomize_targets, dataset_df["target"]
            )

        return dataset_df

    dataset_df = _get_dataframe_from_dataset_folder(
        path_to_dataset=config["dataloader_settings"]["dataset_path"],
        heuristic_for_file_discovery=heuristic_for_file_discovery,
        heuristic_for_subject_ID_extraction=heuristic_for_subject_ID_extraction,
        heuristic_for_class_extraction=heuristic_for_class_extraction,
        heuristic_for_relative_position_extraction=heuristic_for_relative_position_extraction,
    )

    # trim based on requested classes
    if len(config["dataloader_settings"]["classes_of_interest"]) != 0:
        dataset_df = dataset_df[
            dataset_df.target.isin(config["dataloader_settings"]["classes_of_interest"])
        ]
        # print stats
        print(
            f"ATTENTION!!! Using only limited number of classes. Check if this is indented."
        )
        for c in config["dataloader_settings"]["classes_of_interest"]:
            print(
                f"Found {len(dataset_df.loc[dataset_df.target == c])} files to work on for class {c} ({len(pd.unique(dataset_df.loc[dataset_df.target == c].subject_IDs))} unique subjects)."
            )

    # trim number of sample if requested (usially debugging to run faster)
    if config["debugging_settings"]["dataset_fraction"] != 1.0:
        dataset_df = dataset_df.head(
            int(len(dataset_df) * config["debugging_settings"]["dataset_fraction"])
        )

    print(
        f'Found {len(dataset_df)} files to work on ({len(pd.unique(dataset_df["subject_IDs"]))} unique subjects).'
    )

    # trim based on the range of relative positions to include
    if select_slices_in_range:
        dataset_df = _select_slices_in_range(dataset_df, select_slices_in_range)

    if any([randomize_subject_labels, randomize_slice_labels]):
        print(
            f"Randomizing labels: subject-wise:{randomize_subject_labels}, slice-wise: {randomize_slice_labels}"
        )
        dataset_df = _randomize_labels(
            dataset_df,
            randomize_subject_labels,
            randomize_slice_labels,
            seed=config["training_settings"]["random_state"],
        )

    # reset index before splitting, if not the stratified split breaks.
    dataset_df = dataset_df.reset_index()

    # ################## work on splitting
    if config["dataloader_settings"]["class_stratification"]:
        print("Performing stratified data split (on a per subject bases).")
        # Here one needs to have the target for all the classes. Raise error if not
        if not any(dataset_df.target):
            raise ValueError(
                "Requesting stratified split of the data. However, classes could not be infered from the dataset file names."
            )

        # ################## TEST SET
        # perform stratified split
        sgkf = StratifiedGroupKFold(
            n_splits=int(1 / config["dataloader_settings"]["test_size"]),
            shuffle=True,
            random_state=config["training_settings"]["random_state"]
            + repetition_number,
        )

        train_val_ix, test_ix = next(
            sgkf.split(dataset_df, y=dataset_df.target, groups=dataset_df.subject_IDs)
        )

        # get training set
        df_test_split = dataset_df.loc[test_ix].reset_index()
        print(
            f'{"Test set":9s}: {len(test_ix):5d} {"test":10} files ({len(pd.unique(df_test_split["subject_IDs"])):4d} unique subjects ({config["dataloader_settings"]["classes_of_interest"]} {[len(pd.unique(df_test_split.loc[df_test_split.target == c].subject_IDs)) for c in config["dataloader_settings"]["classes_of_interest"]]}))'
        )

        # get train_val set
        df_train_val_split = dataset_df.loc[train_val_ix].reset_index()
        # make a copy of the df_train_val_split to use as back bone for the dataframe to be returned (add the test at the end)
        dataset_split_df = df_train_val_split.copy()

        # ################# TRAINING and VALIDATION SETs
        sgkf = StratifiedGroupKFold(
            n_splits=config["training_settings"]["nbr_inner_cv_folds"]
            if config["training_settings"]["nbr_inner_cv_folds"] != 1
            else int(1 / config["dataloader_settings"]["test_size"]) - 1,
            shuffle=True,
            random_state=config["training_settings"]["random_state"]
            + repetition_number,
        )

        # if only one internal fold is requested, do as in the testing. Else,
        # get all the folds (just use next as many times as the one requested by nbr of internal
        # folds)
        for cv_f, (train_ix, val_ix) in enumerate(
            sgkf.split(
                df_train_val_split,
                groups=df_train_val_split.subject_IDs,
                y=df_train_val_split.target,
            )
        ):
            # add a column in the dataset_split_df and flag all the files based on the split
            dataset_split_df[f"fold_{cv_f+1}"] = "validation"
            # flag the training files
            dataset_split_df.loc[train_ix, f"fold_{cv_f+1}"] = "training"

            # add to the df_test_split the flag for this fold
            df_test_split[f"fold_{cv_f+1}"] = "test"

            # print summary
            aus_df = df_train_val_split.loc[train_ix]
            print(
                f'Fold {cv_f+1:4d}: {len(train_ix):5d} {"training":10} files ({len(pd.unique(df_train_val_split.loc[train_ix]["subject_IDs"])):4d} unique subjects ({config["dataloader_settings"]["classes_of_interest"]} {[len(pd.unique(aus_df.loc[aus_df.target == c].subject_IDs)) for c in config["dataloader_settings"]["classes_of_interest"]]}))'
            )
            aus_df = df_train_val_split.loc[val_ix]
            print(
                f'Fold {cv_f+1:4d}: {len(val_ix):5d} {"validation":10} files ({len(pd.unique(df_train_val_split.loc[val_ix]["subject_IDs"])):4d} unique subjects ({config["dataloader_settings"]["classes_of_interest"]} {[len(pd.unique(aus_df.loc[aus_df.target == c].subject_IDs)) for c in config["dataloader_settings"]["classes_of_interest"]]}))'
            )

            if cv_f + 1 == config["training_settings"]["nbr_inner_cv_folds"]:
                break

    else:
        # create split withouth stratification
        # ################## TEST SET
        gs = GroupShuffleSplit(
            n_splits=1,
            train_size=(1 - config["dataloader_settings"]["test_size"]),
            random_state=config["training_settings"]["random_state"]
            + repetition_number,
        )
        train_val_ix, test_ix = next(
            gs.split(dataset_df, groups=dataset_df.subject_IDs)
        )

        df_test_split = dataset_df.loc[test_ix].reset_index()

        print(
            f'{"Test set":9s}: {len(test_ix):5d} {"test":10} files ({len(pd.unique(df_test_split["subject_IDs"])):4d} unique subjects)'
        )

        # ################## TRAINING and VALIDATION SETs
        df_train_val_split = dataset_df.loc[train_val_ix].reset_index()
        # make a copy of the df_train_val_split to use as back bone for the dataframe to be returned (add the test at the end)
        dataset_split_df = df_train_val_split.copy()

        gs = GroupShuffleSplit(
            n_splits=config["training_settings"]["nbr_inner_cv_folds"],
            train_size=config["data_loader_settings"]["train_val_ratio"]
            if config["training_settings"]["nbr_inner_cv_folds"] == 1
            else (1 - 1 / config["training_settings"]["nbr_inner_cv_folds"]),
            random_state=config["training_settings"]["random_state"]
            + repetition_number,
        )

        for cv_f, (train_ix, val_ix) in enumerate(
            gs.split(df_train_val_split, groups=df_train_val_split.subject_IDs)
        ):
            # add a column in the dataset_split_df and flag all the files based on the split
            dataset_split_df[f"fold_{cv_f+1}"] = "validation"
            # flag the training files
            dataset_split_df.loc[train_ix, f"fold_{cv_f+1}"] = "training"

            # add to the df_test_split the flag for this fold
            df_test_split[f"fold_{cv_f+1}"] = "test"

            print(
                f'Fold {cv_f+1:4d}: {len(train_ix):5d} {"training":10} files ({len(pd.unique(df_train_val_split.loc[train_ix]["subject_IDs"])):4d} unique subjects)'
            )
            print(
                f'Fold {cv_f+1:4d}: {len(val_ix):5d} {"validation":10} files ({len(pd.unique(df_train_val_split.loc[val_ix]["subject_IDs"])):4d} unique subjects)'
            )

    # finish up the dataset_split_df by merging the df_test_split
    dataset_split_df = pd.concat(
        [dataset_split_df, df_test_split], ignore_index=True
    ).reset_index(drop=True)

    # save infromation as csv
    dataset_split_df.to_csv(
        os.path.join(
            config["logging_settings"]["checkpoint_path"], "data_split_information.csv"
        ),
        index=False,
    )

    return dataset_split_df


class PNGDatasetFromFolder(Dataset):
    def __init__(
        self,
        item_list,
        transform,
        labels=None,
        return_file_path=False,
        return_age: bool = False,
        ages=None,
    ):
        super().__init__()
        self.item_list = item_list
        self.nbr_total_imgs = len(self.item_list)
        self.transform = transform
        self.labels = labels
        self.return_file_path = return_file_path
        self.return_age = return_age
        self.ages = ages

    def __len__(
        self,
    ):
        return self.nbr_total_imgs

    def __getitem__(self, item_index):
        item_path = self.item_list[item_index]
        image = PIL.Image.open(item_path).convert("RGB")
        if self.transform:
            tensor_image = self.transform(image)
        else:
            # just convert to tensor
            tensor_image = T.functional.pil_to_tensor(image)
        # compose what needs to be returned
        label, age, item_path = 0, 0, 0

        if self.labels:
            label = self.labels[item_index]
        if self.return_file_path:
            item_path = item_path

        # return label as well
        if self.labels:
            label = self.labels[item_index]
            if self.return_file_path:
                return tensor_image, label, item_path
            else:
                return tensor_image, label
        else:
            if self.return_file_path:
                return tensor_image, 0, item_path
            else:
                return tensor_image, 0


class CustomDataset(pl.LightningDataModule):
    def __init__(
        self,
        train_sample_paths,
        batch_size,
        num_workers,
        validation_sample_paths=None,
        train_val_ratio=0.8,
        test_sample_paths=None,
        preprocess=None,
        transforms=None,
        training_targets=None,
        validation_targets=None,
        test_targets=None,
        # heuristic_for_class_extraction=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_sample_paths = train_sample_paths
        self.validation_sample_paths = validation_sample_paths
        self.test_sample_paths = test_sample_paths
        if isinstance(self.test_sample_paths, (list, tuple)):
            self.return_test_dataloader = True
            print("Returning testing dataloader")
        else:
            self.return_test_dataloader = False
        self.preprocess = preprocess
        self.transform = transforms
        self.num_workers = num_workers
        self.train_val_ratio = train_val_ratio
        self.return_classes = True if training_targets else False
        # TODO make it work even when only train_sample_paths is given

        # get classes
        if self.return_classes:
            # self.train_per_sample_target_class = [
            #     heuristic_for_class_extraction(str(f)) for f in self.train_sample_paths
            # ]
            # self.validation_per_sample_target_class = [
            #     heuristic_for_class_extraction(str(f))
            #     for f in self.validation_sample_paths
            # ]
            # if self.return_test_dataloader:
            #     self.test_per_sample_target_class = [
            #         heuristic_for_class_extraction(str(f))
            #         for f in self.test_sample_paths
            #     ]

            self.train_per_sample_target_class = training_targets
            self.validation_per_sample_target_class = validation_targets
            if self.return_test_dataloader:
                self.test_per_sample_target_class = test_targets

            # get unique target classes
            unique_targe_classes = list(
                dict.fromkeys(self.train_per_sample_target_class)
            )
            unique_targe_classes.sort()
            unique_targe_classes = list(dict.fromkeys(unique_targe_classes))
            one_hot_encodings = torch.nn.functional.one_hot(
                torch.tensor(list(range(len(unique_targe_classes))))
            )
            one_hot_encodings = [i.type(torch.float32) for i in one_hot_encodings]
            # build mapping between class and one hot encoding
            self.target_class_to_one_hot_mapping = dict(
                zip(unique_targe_classes, one_hot_encodings)
            )

            # make one hot encodings
            self.train_per_sample_target_class = [
                self.target_class_to_one_hot_mapping[c]
                for c in self.train_per_sample_target_class
            ]
            self.validation_per_sample_target_class = [
                self.target_class_to_one_hot_mapping[c]
                for c in self.validation_per_sample_target_class
            ]
            if self.return_test_dataloader:
                self.test_per_sample_target_class = [
                    self.target_class_to_one_hot_mapping[c]
                    for c in self.test_per_sample_target_class
                ]
        else:
            self.train_per_sample_target_class = None
            self.validation_per_sample_target_class = None
            self.test_per_sample_target_class = None

    def prepare_data(self):
        print("Working on preparing the data")
        return

    def setup(self, stage=None):
        self.training_set = PNGDatasetFromFolder(
            self.train_sample_paths,
            transform=self.transform,
            labels=self.train_per_sample_target_class,
        )
        self.validation_set = PNGDatasetFromFolder(
            self.validation_sample_paths,
            transform=self.preprocess,
            labels=self.validation_per_sample_target_class,
        )

        if self.return_test_dataloader:
            self.test_set = PNGDatasetFromFolder(
                self.test_sample_paths,
                transform=self.preprocess,
                labels=self.test_per_sample_target_class,
            )

    def train_dataloader(self):
        return DataLoader(
            self.training_set,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_set,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        if self.return_test_dataloader:
            return DataLoader(
                self.test_set,
                self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )

    def get_class_weights(self):
        if self.return_classes:
            # compute class weights based on the training set
            training_target_classes = [
                v.numpy().argmax() for v in self.train_per_sample_target_class
            ]
            class_ratios = class_weight.compute_class_weight(
                class_weight="balanced",
                classes=np.unique(training_target_classes),
                y=training_target_classes,
            )
            return class_ratios, self.target_class_to_one_hot_mapping.keys()
        else:
            return None


def _plot_histogram_from_dataloader(dataloader: pl.LightningDataModule, config):
    # plots the histogram of the training, validation and testing sets available in the dataset

    # ## get training validation and, if available, testing sets
    dataloader.setup()
    training_set = dataloader.train_dataloader()
    validation_set = dataloader.val_dataloader()
    if dataloader.test_sample_paths is not None:
        test_set = dataloader.test_dataloader()
    else:
        test_set = None

    # build plot
    for set, set_name in zip(
        [training_set, validation_set, test_set], ["Training", "Validation", "Test"]
    ):
        # get all the elements in the set
        values = []
        set = iter(set)
        for i in range(len(set)):
            x, _ = next(set)
            x = x.numpy()[:, 0, :, :]
            values.append(x.reshape(x.shape[0], -1))
        # stach all elements
        values = np.vstack(values)

        # start plotting
        print(f"Plotting histogram for {set_name} set.")
        fig, axis = plt.subplots(nrows=1, ncols=1)
        for i in range(values.shape[0]):
            print(f"Worning on {i+1}\{values.shape[0]} line. \r", end="")
            # plot a line histogram for each element
            value = values[i, :]
            sns.kdeplot(value, alpha=0.5)
            # density = stats.gaussian_kde(value)
            # num_bins = 50
            # n, x, _ = plt.hist(value, num_bins, density=True, histtype="step")
            # axis.plot(x, density(x), alpha=0.5)
            axis.set_ylim([1e-5, 1e1])
            axis.set_yscale("log")
            # if i == 9:
            #     break

        # save figure
        print("\nSaving figure...")
        fig.savefig(
            os.path.join(config["working_dir"], f"Histogram_{set_name}_set.png")
        )
        print("Done!")
