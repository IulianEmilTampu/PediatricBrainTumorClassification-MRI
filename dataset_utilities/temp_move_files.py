# %%
import os
import shutil
import glob
from pathlib import Path

# %% MOVE BRAIN MASKS INTO DIAGNOSYS FOLDERS
dataset_path = "/flush/iulta54/Research/Data/CBTN/FILTERED_DATASET_20230117"
brain_masks_folder = "/flush/iulta54/Research/Data/CBTN/FILTERED_DATASET_20230117_HDBET"
# move through the diagnose folders
for diagnose in glob.glob(os.path.join(dataset_path, "*")):
    for subject_id in glob.glob(os.path.join(diagnose, "*")):
        for scan_session in glob.glob(os.path.join(subject_id, "*")):
            for file in glob.glob(os.path.join(scan_session, "*.nii.gz")):
                # build name of the brain mask file
                masked_file_name = f"{os.path.basename(file).split('.')[0]}_mask.nii.gz"
                # search the file in the brain_masks folder
                aus_brain_mask_file = os.path.join(brain_masks_folder, masked_file_name)
                if os.path.isfile(aus_brain_mask_file):
                    # copy file in the
                    shutil.copyfile(
                        aus_brain_mask_file,
                        os.path.join(scan_session, masked_file_name),
                    )
                else:
                    print(f"Brain mask not found for file: {os.path.basename(file)}")

# %% STRUCTURE DATA AFTER HD-BET
dataset_path = "/flush/iulta54/Research/Data/CBTN/FILTERED_DATASET_20230117"
brain_masks_folder = "/flush/iulta54/Research/Data/CBTN/FILTERED_DATASET_20230117_HDBET"
# move through the files in the diagnosys folder
for file in glob.glob(os.path.join(dataset_path, "*.nii.gz")):
    # get diagnosys, patient ID and session
    aus_f = os.path.basename(file)
    diagnosis = aus_f.split("_")[0]
    patientID = aus_f.split("_")[1]
    session = "_".join(aus_f.split("_")[2:5])

    # build folder where to save the files
    save_path = os.path.join(dataset_path, diagnosis, patientID, session)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # move the original file
    shutil.copyfile(file, os.path.join(save_path, aus_f))

    # build name of the brain mask file
    masked_file_name = f"{os.path.basename(file).split('.')[0]}_mask.nii.gz"

    # search the file in the brain_masks folder
    aus_brain_mask_file = os.path.join(brain_masks_folder, masked_file_name)
    if os.path.isfile(aus_brain_mask_file):
        # copy file in the
        shutil.copyfile(
            aus_brain_mask_file,
            os.path.join(save_path, masked_file_name),
        )
    else:
        print(f"Brain mask not found for file: {os.path.basename(file)}")

# %% MOVE FILES THAT HAVE PASSED THE MANUAL CHECK
import pandas as pd

FILTER_FILE = "/flush/iulta54/Research/Data/CBTN/CSV_files/checked_files_min_slice_number_50_maxres_1mm_20230116.xlsx"
RAW_DATASET_LOCATION = (
    "/run/media/iulta54/GROUP_HD1/Tamara/Datasets/CBTN01_min10slices_maxresolution3mm"
)
FILTERED_DATASET_SAVE_PATH = (
    "/flush/iulta54/Research/Data/CBTN/FILTERED_DATASET_20230117"
)

# LOAD THE FILTER_FILE AND FILTER FILES THAT DID NOT PASSED MANUAL CHECK
info = pd.read_excel(FILTER_FILE)

# Filter the data frame to only include rows where the value in the 'final_check' column is 'passed'
info = info[info.final_check == "passed"].reset_index(drop=True)

# LOOP THROUGH THE REMAINING (PER MODALITY) AND SAVE SLICES
modalities = ["T1Gd"]
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

    # finally loop through the subject volumes
    for ind in list(aus_info.index):
        # Get the values for the various columns of the current row
        diagnosis = aus_info["diagnosis"][ind]
        subject = aus_info["subject_ID"][ind]
        session = aus_info["session_ID"][ind]
        filename = aus_info["filename"][ind]
        loc = aus_info["location"][ind]

        print(
            f"Working index {ind} diagnose: {diagnosis}  subject: {subject}  session: {session} "
        )

        # get modality and mask volumes, and apply brain mask
        if modality == "T1Gd":
            modality = "t1c"

        volumes = glob.glob(
            os.path.join(RAW_DATASET_LOCATION, diagnosis, subject, session, "*.nii.gz")
        )
        try:
            volume_path = [i for i in volumes if f"{modality.lower()}.nii.gz" in i][0]

            # copy volume to the specified save folder
            shutil.copyfile(
                volume_path,
                os.path.join(FILTERED_DATASET_SAVE_PATH, os.path.basename(volume_path)),
            )
        except:
            print("This volume vas not found")
