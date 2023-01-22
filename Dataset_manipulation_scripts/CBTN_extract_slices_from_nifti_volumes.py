"""
This code contains functions to manipulate medical imaging data stored in NIfTI format. 
The functions perform tasks such as normalizing and isotropically resampling the data, and visualizing the data in various planes.
The first function, plot_volumes, takes a 3D volume and plots it in all three planes (x, y, z). The second function, normalization, 
normalizes the data in the volume by setting the maximum value to a chosen percentile and the minimum value to the remaining percentile. 
The third function, isotropic_resampling, resamples the volume to a specified size and resamples the voxels to a specified size. 
The fourth function, label_extraction, extracts labels from a CSV file and stores them in a dictionary. 
The final function, extract_rois, extracts regions of interest from the volume using the labels and plots them.

Code taken from Tamara Bianchesi (16/01/2023) and modified by Iulian Emil Tampu.

The code is updated to work on the dataset which has the brain masks computed using the HD-BET software.
"""
#%% Importing necessary libraries
import os
import numpy as np
import glob
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import re
import random
from PIL import Image
from pickletools import read_unicodestringnl
import scipy as sp
from pathlib import Path
import nibabel.processing
from collections import Counter
import matplotlib.patches as patches
import argparse

# %% DEFINE LOCAL UTILITIES


def plot_volumes(volume, img_title="MR volume in the three planes"):
    """
    Plot the MR volume in all three planes.
    Input:
    - volume: obtained from the NIfTI file as .get_fdata()
    - img_title: title for the plot "title"
    """
    x, y, z = volume.shape
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(np.rot90(volume[int(x / 2), :, :], 1), cmap="gray")
    ax[1].imshow(np.rot90(volume[:, int(y / 2), :], 1), cmap="gray")
    ax[2].imshow(np.rot90(volume[:, :, int(z / 2)], 1), cmap="gray")
    ax[0].title.set_text("yz plane")
    ax[1].title.set_text("xz plane")
    ax[2].title.set_text("xy plane")
    plt.suptitle(img_title)
    plt.show()


def normalization(volume, percentile, max_norm=255, plot=False):
    """
    Normalize the MR volume by considering the a chosen percentile maximum, the minumum,
    and set the values between a defined max_norm (usually 255) and min (usually 0). Also the normalized
    volume is ploted by default.
    Input:
    - volume: NIfTI volume
    - percentile: chosen percentile to save the maximum of the volume. either 98% or 95&
    - max_norm: maximum value for the normalization
    - plot: if you want to have the plot or not. Default is True.
    Output:
    - Normalized volume in the form of a np.array
    """
    img_data = volume.get_fdata()
    # img_data[img_data<0]=0
    max_volume = np.nanmax(np.nanpercentile(img_data, percentile))
    img_data[img_data > max_volume] = max_volume
    min_volume = np.nanmin(img_data)
    min_volume = np.nanpercentile(img_data, 100 - percentile)
    # min_volume = np.percentile(img_data, 100 - 98)
    # print(max_volume, min_volume)

    img_data[img_data < min_volume] = min_volume

    normalized_volume = (
        max_norm * (img_data - min_volume) / (max_volume - min_volume)
    ).astype(np.uint8)
    print(max_volume, min_volume, normalized_volume.max(), normalized_volume.min())
    if plot == True:
        plot_volumes(normalized_volume, "Normalized volume")
    normalized_nifti = nib.Nifti1Image(normalized_volume, volume.affine, volume.header)
    return normalized_nifti


def apply_brain_mask(volume, brain_mask):
    """
    Utility that given the volume and the brain mask applyes the brain mask and return a
    nibabel object
    """
    return nib.Nifti1Image(
        volume.get_fdata() * brain_mask.get_fdata(), volume.affine, volume.header
    )


def istropic_resampling(
    volume, tr_or, x_out=224, y_out=224, voxel_size=1.0, spline_order=5, plot=False
):
    """
    Function that isotropically resamples the NIfTI volume. The resampled volume is also plot by default.
    Input:
    - volume: volume obtained from nib.load(path_to_the_volume), the file is in the format .nii.gz
    - x_out, y_out, z_out: chosen size of the output image. Default is 240 for all three values since it
    is what is usually required by cnn models.
    - voxel_size: set to 1.0, it has to be the same value for all three dimensions of the voxel since we
    are doing an isotric resampling
    - spline_order: set to 5, it can have any value between 0 and 5.
    - plot: if you want to have the plot or not. Default is True.
    Output:
    Output:
    - resampled_volume returned as a np.array of three dimensions equal to x_out, y_out, z_out
    """
    if tr_or == "z":
        pix = volume.header["pixdim"][3]
        _, _, z = volume.get_fdata().shape
    elif tr_or == "y":
        pix = volume.header["pixdim"][2]
        _, z, _ = volume.get_fdata().shape
    elif tr_or == "x":
        pix = volume.header["pixdim"][0]
        z, _, _ = volume.get_fdata().shape
    perc_diff = (pix - voxel_size) / voxel_size
    z_out = int(z + (perc_diff) * z)
    resampled_volume = nib.processing.conform(
        volume,
        (x_out, y_out, z_out),
        (voxel_size, voxel_size, voxel_size),
        spline_order,
    )
    if plot == True:
        plot_volumes(resampled_volume.get_fdata(), "Isotropically resampled volume")
    return z_out, resampled_volume


def update_tumor_ROI_based_on_orientation(volume, tumor_ROI_info):
    # Get the shape of the data
    x, y, z = volume.get_fdata().shape

    # get original tumor ROI vertexes
    tr_min = int(tumor_ROI_info["tr_min"])
    tr_max = int(tumor_ROI_info["tr_max"])
    sg_min = int(tumor_ROI_info["sg_min"])
    sg_max = int(tumor_ROI_info["sg_max"])
    fr_min = int(tumor_ROI_info["fr_min"])
    fr_max = int(tumor_ROI_info["fr_max"])

    # % just for clarity we want to make sure that the indexes of the tumor ROI match the
    # x, y, z of the anatomical volume. Here we convert the indexes obtained manually through Itksnap
    # to the anatomical volume orientation.
    # Note that, at the end, we want sg to map x, fr to map y and tr tp map z.

    # Determine the orientation of the data based on the affine matrix
    if (nib.aff2axcodes(vol.affine)) == ("L", "A", "S"):
        # fix x
        sg_or = x
        sg_min = int(tumor_ROI_info["sg_min"])
        sg_max = int(tumor_ROI_info["sg_max"])
        # fix y
        fr_or = y
        fr_min = int(tumor_ROI_info["fr_min"])
        fr_max = int(tumor_ROI_info["fr_max"])

        # fix z
        tr_or = z

    elif (nib.aff2axcodes(vol.affine)) == ("L", "S", "P"):
        # invert
        sg_or = x
        sg_min = int(tumor_ROI_info["sg_min"])
        sg_max = int(tumor_ROI_info["sg_max"])

        if (tumor_ROI_info["original_tr"].values)[0] == "y":
            # swap fr with tr
            fr_or = y
            fr_min = int(tumor_ROI_info["tr_min"])
            fr_max = int(tumor_ROI_info["tr_max"])

            tr_or = z
            tr_min = int(tumor_ROI_info["fr_min"])
            tr_max = int(tumor_ROI_info["fr_max"])

        elif (tumor_ROI_info["original_tr"].values)[0] == "z":
            fr_or = z
            fr_min = int(tumor_ROI_info["fr_min"])
            fr_max = int(tumor_ROI_info["fr_max"])

            tr_or = y
            tr_min = int(tumor_ROI_info["tr_min"])
            tr_max = int(tumor_ROI_info["tr_max"])

    elif (nib.aff2axcodes(vol.affine)) == ("P", "S", "R"):
        # fix x
        sg_or = x
        sg_min = int(tumor_ROI_info["fr_min"])
        sg_max = int(tumor_ROI_info["fr_max"])
        # fix y
        fr_or = y
        fr_min = int(tumor_ROI_info["tr_min"])
        fr_max = int(tumor_ROI_info["tr_max"])
        # fix z
        tr_or = z
        tr_min = int(tumor_ROI_info["sg_min"])
        tr_max = int(tumor_ROI_info["sg_max"])

    elif (nib.aff2axcodes(vol.affine)) == ("R", "A", "S"):
        sg_or = x
        sg_min = int(tumor_ROI_info["sg_min"])
        sg_max = int(tumor_ROI_info["sg_max"])

        fr_or = y
        fr_min = int(tumor_ROI_info["fr_min"])
        fr_max = int(tumor_ROI_info["fr_max"])

        tr_or = z

    elif (nib.aff2axcodes(vol.affine)) == ("P", "S", "L"):
        sg_or = z
        sg_min = sg_or - int(tumor_ROI_info["sg_max"])
        sg_max = sg_or - int(tumor_ROI_info["sg_min"])
        fr_or = x
        fr_min = fr_or - int(tumor_ROI_info["fr_max"])
        fr_max = fr_or - int(tumor_ROI_info["fr_min"])
        tr_or = y

    return sg_or, sg_min, sg_max, tr_or, tr_min, tr_max, fr_or, fr_min, fr_max


def get_brain_annotation_indexes(
    anatomical_volume,
    annotation_volume,
    axis_to_slice,
    min_nbr_nonzero_pixels_brain=500,
    min_nbr_nonzero_pixels_annotation=50,
):
    """
    Utility that given the anatomical and annotation volume, returns the indexes along a specified
    axis of the slices that contain brain slices and brain slices that contain tumor.
    """

    # transpose the volumes to make the slice index the first
    if axis_to_slice == 2:
        anatomical_volume = anatomical_volume.transpose((1, 0, 2))
        annotation_volume = annotation_volume.transpose((1, 0, 2))
    elif axis_to_slice == 3:
        anatomical_volume = anatomical_volume.transpose((2, 0, 1))
        annotation_volume = annotation_volume.transpose((2, 0, 1))
    elif axis_to_slice == 1:
        # nothing to do here
        anatomical_volume = anatomical_volume
        annotation_volume = annotation_volume

    brain_slices_indexes = [
        idx
        for idx in range(anatomical_volume.shape[0])
        if len(np.nonzero(anatomical_volume[idx, :, :])[0])
        >= min_nbr_nonzero_pixels_brain
    ]
    brain_slices_indexes = np.array(brain_slices_indexes)

    tumor_slices_indexes = [
        idx
        for idx in range(annotation_volume.shape[0])
        if len(np.nonzero(annotation_volume[idx, :, :])[0])
        >= min_nbr_nonzero_pixels_annotation
    ]
    tumor_slices_indexes = np.array(tumor_slices_indexes)

    return brain_slices_indexes, tumor_slices_indexes


def slice_volume_and_save(
    anatomical_vol,
    annotation_vol,
    axis_to_slice,
    brain_slices_indexes,
    tumor_slices_indexes,
    name_prefix,
    save_path,
    img_format,
    save_annotation=False,
):
    """
    Utility that given a couple of anatomical volume with the corresponding annotation
    saves the brain slices with in the name label 0 if there is NO tumor and label 1
    if THERE IS tumor.
    It does not save background images.
    One can also specify which axix to slice along.

    STEPS
    for each slice in the brain (looping though the selected axis)
    1 - check if the slice is not background
        1.1 - if is not, check if the annotation is not all 0s
            1.1.1 - if not, save the image with label 1 (save in the
                    name also the relative position within the tumor)
            1.1.2 - if true, save the image with label 0 (save in the name
                    also the relative distance from the first tumor slice)
    """
    # transpose the volumes to make the slice index the first
    if axis_to_slice == 2:
        anatomical_vol = anatomical_vol.transpose((1, 0, 2))
        anatomical_vol = annotation_vol.transpose((1, 0, 2))
    elif axis_to_slice == 3:
        anatomical_vol = anatomical_vol.transpose((2, 0, 1))
        annotation_volume = annotation_vol.transpose((2, 0, 1))
    elif axis_to_slice == 1:
        # nothing to do here
        anatomical_vol = anatomical_vol
        annotation_volume = annotation_vol

    # print(anatomical_vol.shape)
    # print(annotation_vol.shape)

    # brain_indexes = np.nonzero(np.sum(anatomical_vol, axis=(1, 2)) >= 500)[0]
    # tumor_indexes = np.nonzero(np.sum(annotation_vol, axis=(1, 2)))[0]

    # print(brain_indexes)
    # print(tumor_indexes)

    counter = 0
    for bi in brain_slices_indexes:
        # check if the slice has tumor
        if bi in tumor_slices_indexes:
            # this is a tumor slice
            label = 1
            relative_position = (
                (bi - tumor_slices_indexes.min())
                / (tumor_slices_indexes.max() - tumor_slices_indexes.min())
                * 100
            )
            file_name = f"{name_prefix}_rlp_{relative_position:0.1f}_label_{label}"
            if save_annotation:
                I = annotation_vol[bi, :, :].squeeze()
                I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
                im = Image.fromarray(I8)
                im.save(
                    os.path.join(save_path, f"{file_name}_annotation_.{img_format}")
                )
        else:
            # save brain slide
            label = 0
            relative_position = np.min(np.abs(tumor_slices_indexes - bi))
            file_name = f"{name_prefix}_rlp_{relative_position:0.1f}_label_{label}"

        # save slide
        I = anatomical_vol[bi, :, :].squeeze()
        I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
        im = Image.fromarray(I8)
        im.save(os.path.join(save_path, f"{file_name}.{img_format}"))
        counter += 1

    return counter


# %% parse input

su_debug_flag = True

# --------------------------------------
# read the input arguments and set the base folder
# --------------------------------------
if not su_debug_flag:
    parser = argparse.ArgumentParser(
        description="Script that operates on the CBTN dataset (with brain masks) to extract transversal slices."
    )
    parser.add_argument(
        "-dataset_folder",
        "--DATASET_FOLDER",
        required=True,
        type=str,
        help="Path to the folder where the .nifti files of the subjects are stored. Here we expect the files to be saved as Dignosis/SubjectID/ScanID/*.nii.gz",
    )
    parser.add_argument(
        "-filter_file",
        "--FILTER_FILE",
        required=True,
        type=str,
        help="Path to the .xlsx file which is used to filter the dataset.",
    )
    parser.add_argument(
        "-save_path",
        "--SAVE_PATH",
        required=False,
        type=str,
        default=None,
        help="Provide the path to where the extracted slices will be saved. By default will be saved in the parent of the DATASET_FOLDER under EXTRACTED_SLICES. Slices from each modality will be saved in sevarate folders within the EXTRACTED_SLICES folder.",
    )
    parser.add_argument(
        "-slice_shape",
        "--SLICE_SHAPE",
        required=False,
        type=tuple,
        default=(240, 240),
        help="Specify the size of the output slices (used during isotropical interpolation).",
    )
    # other parameters
    parser.add_argument(
        "-rns",
        "--RANDOM_SEED_NUMBER",
        required=False,
        type=int,
        default=29122009,
        help="Specify random number seed.",
    )
    args_dict = dict(vars(parser.parse_args()))

else:
    # # # # # # # # # # # # # # DEBUG
    args_dict = {
        "DATASET_FOLDER": "/flush/iulta54/Research/Data/CBTN/FILTERED_DATASET_20230117",
        "FILTER_FILE": "/flush/iulta54/Research/Data/CBTN/CSV_files/checked_files_min_slice_number_50_maxres_1mm_20230116.xlsx",
        "SAVE_PATH": "/flush/iulta54/Research/Data/CBTN/EXTRACTED_SLICES",
        "SLICE_SHAPE": (224, 224),
        "RANDOM_SEED_NUMBER": 29122009,
    }

# check if the DATASET_FOLDER exists if given.
if args_dict["DATASET_FOLDER"]:
    if not os.path.isdir(args_dict["DATASET_FOLDER"]):
        raise ValueError(
            f"The given DATASET_FOLDER does not exist. Provide a valid one."
        )

# check if the DATASET_FOLDER exists if given.
if args_dict["FILTER_FILE"]:
    if not os.path.isfile(args_dict["FILTER_FILE"]):
        raise ValueError(f"The given FILTER_FILE does not exist. Provide a valid one.")

# check if the SAVE_PATH exists if given. Else create one
if args_dict["SAVE_PATH"]:
    if not os.path.isdir(args_dict["SAVE_PATH"]):
        raise ValueError(
            f"The given SAVE_PATH does not exist. Provide a valid one or None (in this case one will be created)"
        )
else:
    args_dict["SAVE_PATH"] = os.path.join(
        os.path.dirname(args_dict["DATASET_FOLDER"]), "EXTRACTED_SLICES"
    )
    Path(args_dict["SAVE_PATH"]).mkdir(parents=True, exist_ok=True)

# print input variables
max_len = max([len(key) for key in args_dict])
[print(f"{key:{max_len}s}: {value}") for key, value in args_dict.items()]

#%% LOAD THE FILTER_FILE AND FILTER FILES THAT DID NOT PASSED MANUAL CHECK
info = pd.read_excel(args_dict["FILTER_FILE"])

# Filter the data frame to only include rows where the value in the 'final_check' column is 'passed'
info = info[info.final_check == "passed"].reset_index(drop=True)

# %% LOOP THROUGH THE REMAINING (PER MODALITY) AND SAVE SLICES
modalities = ["T2", "T1"]
processed_indexes = dict.fromkeys(modalities)
total_slice_counter = 0
for modality in modalities:
    print(f"Working on {modality} modality...")
    # filter based on modality
    aus_info = info[info.actual_modality == modality].reset_index(drop=True)

    # If the 'modality' variable is 'T1Gd', perform additional filtering on the data frame
    if modality == "T1Gd":
        # Create two lists of subject-session combinations from the 'subject_session' column
        volumes_t1 = list(
            aus_info[aus_info.original_modality == "T1"]
            .reset_index(drop=True)
            .subject_session
        )
        volumes_t1c = list(
            aus_info[aus_info.original_modality == "T1Gd"]
            .reset_index(drop=True)
            .subject_session
        )
        # Find the intersection of the two lists to get a list of subject-session combinations that appear in both lists
        double_volumes = list(set(volumes_t1c) & set(volumes_t1))
        # Remove the rows corresponding to the subject-session combinations in the 'double_volumes' list from the data frame
        lose = []
        for v in double_volumes:
            idx = aus_info.loc[aus_info["subject_session"] == v].index.to_list()
            i1 = idx[0]
            i2 = idx[1]
            lose.append(i1)
        aus_info = aus_info.drop(lose)

    # create folder for this modality and clean all if anything inside already
    modality_folder = os.path.join(args_dict["SAVE_PATH"], modality)
    Path(modality_folder).mkdir(parents=True, exist_ok=True)

    # for root, dirs, files in os.walk(modality_folder):
    #     for file in files:
    #         os.remove(os.path.join(root, file))

    # initialize ausiliary variables
    # lists for storing image data and metadata
    images_file = []
    training_set = []
    testing_set = []
    roi_box = []
    # counting and storing data
    tot = 0
    tumor_roi_tr = []
    ctr = 0

    processed_indexes[modality] = []
    count = 0
    # finally loop through the subject volumes
    for ind in list(aus_info.index):
        # Get the values for the various columns of the current row
        diagnosis = aus_info["diagnosis"][ind]
        subject = aus_info["subject_ID"][ind]
        session = aus_info["session_ID"][ind]
        filename = aus_info["filename"][ind]
        loc = aus_info["location"][ind]
        original_tr = aus_info["original_tr"][ind]

        print(
            f"  Working index {ind}:\n  diagnose: {diagnosis}\n  subject: {subject}\n  session: {session}\n  original_tr: {original_tr} "
        )

        # get modality and mask volumes, and apply brain mask
        volumes = glob.glob(
            os.path.join(
                args_dict["DATASET_FOLDER"], diagnosis, subject, session, "*.nii.gz"
            )
        )
        volume_path = [i for i in volumes if f"{modality}.nii.gz" in i][0]
        mask_path = [i for i in volumes if f"{modality}_mask.nii.gz" in i][0]

        # Load the NIfTI file
        vol = nib.load(volume_path)
        # load the brain mask as well
        brain_mask = nib.load(mask_path)
        # apply brain mask to volume
        vol = apply_brain_mask(vol, brain_mask)

        # update tumor ROI based on volume orientation
        (
            sg_or,
            sg_min,
            sg_max,
            tr_or,
            tr_min,
            tr_max,
            fr_or,
            fr_min,
            fr_max,
        ) = update_tumor_ROI_based_on_orientation(vol, aus_info.iloc[[ind]])

        # build mask of the tumor ROI and save as nifti file (if not present)
        tumor_ROI = [i for i in volumes if f"{modality}_tumor_ROI.nii.gz" in i]
        # if not any(tumor_ROI):
        # build volume
        ROI_volume = np.zeros_like(brain_mask.get_fdata())
        # set to one the value in the slices
        # ROI_volume[sg_min:sg_max, fr_min:fr_max, tr_min:tr_max] = 1

        # debug tumor ROI
        try:
            ROI_volume[sg_min:sg_max, fr_min:fr_max, tr_min:tr_max] = np.ones(
                (sg_max - sg_min, fr_max - fr_min, tr_max - tr_min)
            )
        except:
            print(f"Sum of tumor ROI region: {ROI_volume.sum()}")
            print(
                f"Sum of expected tumor ROI region: {np.sum((sg_max - sg_min, fr_max - fr_min, tr_max - tr_min))}"
            )
            # get original vertexes
            or_tr_min = int(aus_info.iloc[[ind]]["tr_min"])
            or_tr_max = int(aus_info.iloc[[ind]]["tr_max"])
            or_sg_min = int(aus_info.iloc[[ind]]["sg_min"])
            or_sg_max = int(aus_info.iloc[[ind]]["sg_max"])
            or_fr_min = int(aus_info.iloc[[ind]]["fr_min"])
            or_fr_max = int(aus_info.iloc[[ind]]["fr_max"])
            print(
                f"Original vertexes (sg_min, sg_max, fr_min, fr_max, tr_min, tr_max):\n{or_sg_min, or_sg_max, or_fr_min, or_fr_max, or_tr_min, or_tr_max}"
            )
            print(
                f"Updated vertexes (sg_min, sg_max, fr_min, fr_max, tr_min, tr_max):\n{sg_min, sg_max, fr_min, fr_max, tr_min, tr_max}"
            )
            print(f"Tumor ROI nifti volume shape: {ROI_volume.shape}")
            print(f"Image volume orientation : {(nib.aff2axcodes(vol.affine))}")

            raise ValueError(
                f"Something went wrong in the creation of the tumor ROI.\nPlease check!"
            )
        # convert to nifi image
        nifit_ROI_volume = nib.Nifti1Image(
            ROI_volume, brain_mask.affine, brain_mask.header
        )
        # build file name
        aus_file_name = (
            "_".join(os.path.basename(mask_path).split(".")[0].split("_")[0:-1])
            + "_tumor_ROI.nii.gz"
        )
        aus_file_name = os.path.join(
            args_dict["DATASET_FOLDER"], diagnosis, subject, session, aus_file_name
        )
        nib.save(nifit_ROI_volume, aus_file_name)

        print(f" Saving tumor ROI as nifti at:\n{aus_file_name}")

        # normalize the volume using the specified percentile
        print(" Intensity normalization...")
        norm_nifti_vol = normalization(
            vol,
            99.8,
        )

        # resample the volume (and tumor ROI) to isotropic resolution
        x_out, y_out = args_dict["SLICE_SHAPE"][0], args_dict["SLICE_SHAPE"][1]
        print(" Isotropical resampling anatomical volume...")
        z_out, res_nifti = istropic_resampling(
            norm_nifti_vol,
            tr_or=original_tr,
            x_out=x_out,
            y_out=y_out,
            voxel_size=1.0,
            spline_order=5,
        )
        print(" Isotropical resampling tumor annotation...")
        _, tumor_ROI_res_nifti = istropic_resampling(
            nifit_ROI_volume,
            tr_or=original_tr,
            x_out=x_out,
            y_out=y_out,
            voxel_size=1.0,
            spline_order=5,
        )

        # get the data from the resampled volume
        res_vol = res_nifti.get_fdata()
        tumor_ROI = tumor_ROI_res_nifti.get_fdata()
        # remove interpolation artefacts on the annotation
        tumor_ROI[tumor_ROI >= 0.5] = 1
        tumor_ROI[tumor_ROI < 0.5] = 0

        print(f" Anatomical volume size: {res_nifti.shape}")
        print(f" Tumor annotation size: {tumor_ROI_res_nifti.shape}")

        # get slices to save
        print(" Saving slices...")
        print()
        brain_slices_indexes, tumor_slice_indexes = get_brain_annotation_indexes(
            res_vol,
            tumor_ROI,
            axis_to_slice=3,
            min_nbr_nonzero_pixels_brain=2000,
            min_nbr_nonzero_pixels_annotation=10,
        )
        print(f" Brain slice indexes: {brain_slices_indexes[0:10]}...")
        print(f" Tumor slice indexes: {tumor_slice_indexes[0:10]}...")
        # save slices
        name_prefix = os.path.basename(volume_path).split(".")[0]
        slice_counter = slice_volume_and_save(
            anatomical_vol=res_vol,
            annotation_vol=tumor_ROI,
            axis_to_slice=3,
            brain_slices_indexes=brain_slices_indexes,
            tumor_slices_indexes=tumor_slice_indexes,
            name_prefix=name_prefix,
            save_path=modality_folder,
            img_format="png",
            save_annotation=False,
        )
        print(f" Saved {slice_counter} slices from this volume...")
        # save index sample processed
        processed_indexes[modality].append(ind)
        total_slice_counter += slice_counter


print(f"Saved {total_slice_counter} slices!")
