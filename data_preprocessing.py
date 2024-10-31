# %%
"""
Script that given the summary image file folders where the visual check was performed, 
gathers the information of the files that can be used, takes the nifti files from the raw CBTN_v2 dataset and:
 1 - isotropic resamples them
 2 - performs intensity normalization 
 3 - saved renamed file for brain extraction

 Note that the way the summary files were saved and the summary files that are moved due to pre post contrast, makes the identification 
 of the files to use slightly more convoluted than it needs to be. This can be improved...but as for now need to make progress :) 
"""

import os
import sys
import pathlib
import numpy as np
import pandas as pd
import glob
import time
from datetime import datetime
import nibabel as nib
import shutil
import subprocess
import matplotlib.pyplot as plt

import SimpleITK as sitk

import torchio as tio

# %%
"""
The preprocessing should include:
- bias field correction (N4ITK algorithm)
- isotropical resampling (sitkBSpline interpolation)
- skull stripping
- intensity normalization using Z-score 

https://doi.org/10.1038/s41598-020-69298-z
"""


# %% UTILITIES
class SingleMRIModalityPreProcessing:
    def __init__(
        self,
        volume_path,
        save_path,
        save_volume_name,
        output_spacing=(1, 1, 1),
        output_size=(240, 240, 155),
        interpolation="linear",
    ):
        self.volume_path = volume_path
        self.save_path = save_path
        self.save_volume_name = save_volume_name
        self.output_spacing = output_spacing
        self.output_size = output_size
        self.interpolation = interpolation

        self.raw_affine = nib.load(volume_path).affine

    def bias_field_correction(self):
        # https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html
        try:
            # load volume
            raw_img_sitk = sitk.ReadImage(self.volume_path, sitk.sitkFloat32)

            # create mask of the head (reduces computation of the bias fiels correction)
            transformed = sitk.RescaleIntensity(raw_img_sitk, 0, 255)  # 1
            transformed = sitk.LiThreshold(transformed, 0, 1)  # 2
            head_mask = transformed

            # reduce image size to make computation faster (then bring back)
            shrinkFactor = 4
            inputImage = raw_img_sitk

            inputImage = sitk.Shrink(
                raw_img_sitk, [shrinkFactor] * inputImage.GetDimension()
            )
            maskImage = sitk.Shrink(
                head_mask, [shrinkFactor] * inputImage.GetDimension()
            )

            # get bias field correction
            bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrected = bias_corrector.Execute(inputImage, maskImage)

            # bring back image to its original size
            log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img_sitk)
            corrected_image_full_resolution = raw_img_sitk / sitk.Exp(log_bias_field)

            corrected_image_full_resolution.CopyInformation(raw_img_sitk)
            return (True, corrected_image_full_resolution)
        except:
            return (False, "bias_field_correction_failed")

    def reorient_resample_crop(self, simpleITK_Image):
        try:
            # convert image for tio
            subject = tio.Subject(
                image=tio.ScalarImage.from_sitk(simpleITK_Image),
            )

            preprocessor = tio.Compose(
                [
                    tio.ToCanonical(),
                    tio.Resample(
                        target=self.output_spacing,
                        image_interpolation=self.interpolation,
                    ),
                    tio.CropOrPad(self.output_size),
                ]
            )
            subject = preprocessor(subject)

            # return as np volume
            return (True, subject.image.numpy().squeeze(axis=0))
        except:
            return (False, "reorient_resample_crop_failed")

    def get_brain_mask(self, resampled_volume_path):
        try:
            # this uses the HD-BET deep learnin brain extraction tool
            brain_volume_path = os.path.join(
                self.save_path,
                os.path.basename(self.volume_path),
            )
            brain_mask_volume_path = os.path.join(
                self.save_path,
                os.path.basename(self.volume_path).split(".nii.gz")[0] + "_mask.nii.gz",
            )
            stdout = subprocess.run(
                f"conda run -n HD-BET hd-bet -i {resampled_volume_path} -o {brain_volume_path}",
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                check=True,
                text=True,
            ).stdout

            # check if there were any errors
            if "ERROR" in stdout:
                return (False, "brain_extraction_failed")
            else:
                return (True, brain_volume_path, brain_mask_volume_path)
        except:
            return (False, "brain_extraction_failed")

    def z_normalization(self, brain_only_volume_path, brain_mask_path):
        try:
            subject = tio.Subject(
                image=tio.ScalarImage(brain_only_volume_path),
                brain=tio.LabelMap(brain_mask_path),
            )
            transform = tio.ZNormalization(masking_method="brain")
            norm_subject = transform(subject)

            return (True, norm_subject.image.numpy().squeeze(axis=0))
        except:
            return (False, "Zscore_normalization_failed")

    def save_volume_to_nifti(self, np_volume, volume_name):
        """
        Small utility for saving a volume to nifti
        """
        try:
            ni_img = nib.Nifti1Image(np_volume, affine=np.eye(4))
            save_path = os.path.join(self.save_path, volume_name)
            nib.save(ni_img, save_path)
            return (True, save_path)
        except:
            return (False, "save_volume_to_nifti_failed")

    def apply_preprocessing(self):
        # apply bias field correction
        bias_field_correction_output = self.bias_field_correction()

        # apply reorientation to canonical, resample and cropping
        if bias_field_correction_output[0]:
            resampled_output = self.reorient_resample_crop(
                bias_field_correction_output[1]
            )
        else:
            return bias_field_correction_output

        # apply brain extration
        # # save image to nii.gz so that it can be fed to hd-bet
        if resampled_output[0]:
            save_volume_output = self.save_volume_to_nifti(
                resampled_output[1],
                volume_name="RESAMPLED_" + self.save_volume_name,
            )
        else:
            return resampled_output

        # # run HD-BET
        if save_volume_output[0]:
            brain_masking_output = self.get_brain_mask(save_volume_output[1])
        else:
            return save_volume_output

        # apply z-score normalization
        if brain_masking_output[0]:
            zscore_normalization_output = self.z_normalization(
                brain_only_volume_path=brain_masking_output[1],
                brain_mask_path=brain_masking_output[2],
            )
        else:
            return brain_masking_output

        # save normalized volume
        if zscore_normalization_output[0]:
            save_volume_output = self.save_volume_to_nifti(
                zscore_normalization_output[1],
                volume_name="PRE_PROCESSED_" + self.save_volume_name,
            )
        else:
            return zscore_normalization_output

        try:
            # some clean up (remove the not normalized brain only volume, move the resampled volume, the brain mask to different folders)
            # # remove not normalized brain only volume
            os.remove(brain_masking_output[1])
            # # move the resampled version to a different folder
            move_folder = os.path.join(
                self.save_path, "BIAS_FIELD_CORRECTED_AND_RESAMPLED_VOLUMES"
            )
            pathlib.Path(move_folder).mkdir(parents=True, exist_ok=True)
            shutil.move(
                os.path.join(self.save_path, "RESAMPLED_" + self.save_volume_name),
                os.path.join(move_folder, self.save_volume_name),
            )
            # # move brain mask to different folder
            move_folder = os.path.join(self.save_path, "BRAIN_MASKS")
            pathlib.Path(move_folder).mkdir(parents=True, exist_ok=True)
            shutil.move(
                brain_masking_output[2],
                os.path.join(move_folder, self.save_volume_name),
            )

            return (
                True,
                "preprocessing_passed",
                save_volume_output[1],
            )
        except:
            return (False, "clean_up_failed")

    def get_original_resolution(self):
        # load volume
        image = nib.load(self.volume_path)
        return image.header["pixdim"][1:4]


def get_diagnosis_folder(base_dataset_path, subjectID):
    for f in glob.glob(os.path.join(base_dataset_path, "*")):
        if os.path.isdir(os.path.join(f, "SUBJECTS", subjectID)):
            return os.path.basename(f)


# %% DEFINE PATH

SUMMARY_FILES_RAW = (
    "/flush/iulta54/Research/P9-Cross_modal_data_fusion/not_for_git/SUMMARY_IMAGES"
)
SUMMARY_FILES_CORRECTIONS = "/flush/iulta54/Research/P9-Cross_modal_data_fusion/not_for_git/SUMMARY_FILES_CORRECTIONS/MODALITIES"

PATH_TO_SUMMARY_CSV = "/run/media/iulta54/Expansion1/Datasets/CBTN_v2/SUMMARY_FILES/radiology_per_file_scraped_data.csv"
BASE_DATASET_PATH = "/run/media/iulta54/Expansion1/Datasets/CBTN_v2/RADIOLOGY"

date_time = datetime.now()
d = date_time.strftime("t%H%M_d%m%d%Y")
SAVE_PATH = f"/run/media/iulta54/Expansion1/Datasets/CBTN_v2/RADIOLOGY_PER_MODALITY_FOR_PRE_PROCESSING_{d}"
pathlib.Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

# %% OPEN SUMMARY FILE
df_summary_csv = pd.read_csv(PATH_TO_SUMMARY_CSV, low_memory=False)

# FILTER TO ONLY USE PRE-OPERATIVE MR INFORMATION
df_summary_csv = df_summary_csv[df_summary_csv["session_status"] == "pre_op"]

# %% DEFINE WHICH MODELITIES TO WORK ON
MR_SEQUENCES_TO_GROUP = {
    "T1W": ["T1W", "T1W_SE", "T1W_FL", "T1W_MPRAGE", "T1W_FSPGR"],
    "T1W_GD": ["T1W_GD", "T1W_SE_GD", "T1W_FL_GD", "T1W_MPRAGE_GD"],
    # "T2W": ["T2W", "T2W_TRIM"],
    # "FLAIR": ["FLAIR", "T2W_FLAIR"],
}

MR_SEQUENCES_TO_CHECK_FOR_CORRECTIONS = {
    "T1W": (
        ["T1W_GD", "T1W_SE_GD", "T1W_FL_GD", "T1W_MPRAGE_GD"],
        ["9-WITHOUT_CONTRAST"],
    ),
    "T1W_GD": (
        ["T1W", "T1W_SE", "T1W_FL", "T1W_MPRAGE", "T1W_FSPGR"],
        ["6-WITH_CONTRAST"],
    ),
}

MR_SEQUENCES_OF_INTEREST = [
    "T1W",
    "T1W_GD",
]

# %% GET THE LIST OF FILES THAT SHOULD BE PREPROCESSED
# COMBINE THE INFORMATION FROM THE SUMMARY_FILES_RAW, SUMMARY_FILES_CORRECTIONS, PATH_TO_SUMMARY_CSV

PER_MODALITY_PATHS = {}
PER_MODALITY_PATHS_TO_CHECK = {}

for mr_sequence in MR_SEQUENCES_OF_INTEREST:
    # loop through the mr sequences that belong to this group
    for mr_sequence_from_group in MR_SEQUENCES_TO_GROUP[mr_sequence]:
        # get files from SUMMARY_FILES_RAW
        files_to_use = glob.glob(
            os.path.join(SUMMARY_FILES_RAW, mr_sequence_from_group, "*.png")
        )
    # get files from SUMMARY_FILES_CORRECTIONS (use the MR_SEQUENCES_TO_CHECK_FOR_CORRECTIONS)
    for mr_sequence_from_corrections in MR_SEQUENCES_TO_CHECK_FOR_CORRECTIONS[
        mr_sequence
    ][0]:
        for where_to_check in MR_SEQUENCES_TO_CHECK_FOR_CORRECTIONS[mr_sequence][1]:
            # file files
            files_to_use.extend(
                glob.glob(
                    os.path.join(
                        SUMMARY_FILES_CORRECTIONS,
                        mr_sequence_from_corrections,
                        where_to_check,
                        "*.png",
                    )
                )
            )

    # use the df_summary_csv to get the path to the nifti files which created the summary files
    # this the template of the summary file name f"{modality}---{subjectID}---{session_name}---{os.path.basename(file_path).split('.nii.gz')[0]}-{str(file_idx+1)}.png"
    subject_IDs = [os.path.basename(f).split("---")[1] for f in files_to_use]
    session_names = [os.path.basename(f).split("---")[2] for f in files_to_use]
    file_names = [
        "-".join(os.path.basename(f).split("---")[-1].split("-")[0:-1]).split(".png")[0]
        + ".nii.gz"
        for f in files_to_use
    ]
    diagnosis_folder = [
        get_diagnosis_folder(BASE_DATASET_PATH, subj) for subj in subject_IDs
    ]

    # create path to the nifti file
    PER_MODALITY_PATHS[mr_sequence] = []
    PER_MODALITY_PATHS_TO_CHECK[mr_sequence] = []

    for d, s, ss, fn in zip(diagnosis_folder, subject_IDs, session_names, file_names):
        try:
            PER_MODALITY_PATHS[mr_sequence].append(
                (
                    glob.glob(
                        os.path.join(
                            BASE_DATASET_PATH,
                            d,
                            "SUBJECTS",
                            s,
                            "SESSIONS",
                            ss,
                            "ACQUISITIONS",
                            "*",
                            "FILES",
                            fn,
                        )
                    )[0],
                    "___".join([s, ss, fn]),
                )
            )
        except:
            PER_MODALITY_PATHS_TO_CHECK[mr_sequence].append(
                f"TO_CHECK:{d}_{s}_{ss}_{fn}"
            )

# %% PERFORM PRE PROCESSING ON ALL THE FILES

log_preprocessing = {"file_path": [], "pre_processing_passed": [], "status": []}
start_preprocessing_time = time.time()
start_file_preprocessing_time = time.time()

for modality, files_to_process in PER_MODALITY_PATHS.items():
    # create folder where to save the modality pre processed volumes
    MODALITY_SAVE_PATH = os.path.join(SAVE_PATH, modality)
    MODALITY_SUMMARY_FILE_PATH = os.path.join(SAVE_PATH, modality, "SUMMARY_FILES")
    pathlib.Path(MODALITY_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(MODALITY_SUMMARY_FILE_PATH).mkdir(parents=True, exist_ok=True)

    # loop thrugh the different files
    for idx, file_info in enumerate(files_to_process):
        print(
            f"Preprocessing modality {modality} ({idx+1:04d}/{len(files_to_process)}). Last took {time.time() - start_file_preprocessing_time:2.2f} s. Total time {(time.time() - start_preprocessing_time) / 60:.3f} minutes.\r",
            end="",
        )
        start_file_preprocessing_time = time.time()
        preprocessor = SingleMRIModalityPreProcessing(
            volume_path=file_info[0],
            save_path=MODALITY_SAVE_PATH,
            save_volume_name=file_info[1],
        )
        # apply preprocessing
        preprocessing_result = preprocessor.apply_preprocessing()
        status = preprocessing_result[1]

        # save summary image
        if preprocessing_result[0]:
            try:
                subject = tio.Subject(
                    image=tio.ScalarImage(preprocessing_result[2]),
                )
                figure_name = os.path.join(
                    MODALITY_SUMMARY_FILE_PATH,
                    file_info[1].split(".nii.gz")[0] + ".jpeg",
                )
                subject.plot(output_path=figure_name, show=False, figsize=(25, 9))
                plt.close()
            except:
                status = "print_summary_image_failed"

        # save pre processing status for this file
        log_preprocessing["file_path"].append(file_info[0])
        log_preprocessing["pre_processing_passed"].append(preprocessing_result[0])
        log_preprocessing["status"].append(status)

        # save summary every 5 files
        if len(log_preprocessing["file_path"]) % 5 == 0:
            print("Saving preprocessing summary.\n")
            df = pd.DataFrame(log_preprocessing)
            df.to_csv(
                os.path.join(SAVE_PATH, f"{modality}_preprocessing_summary.csv"),
                index_label=True,
            )
