# %%
"""
Script that given the path to the CBTM radiology dataset, scrapes the dataset and saves the information into a CSV file.
## Radiology

After scraping, Should to be able to answer the following questions:

- [ ]  how many patients (male, female, age)
- [ ]  how many sessions (pre-op)
- [ ]  how many T1w, T1wc, T2w, FLAIR and Diffusion volumes we have per subject and per session
- [ ]  what type of diffusion data is available
- [ ]  what mean resolution does the different MR sequences have
- [ ]  from how many scanners was the data collected

The ultimate way to store the information is in a database that we can query. But is might take time, so for now we do it the usual Excel way. Several Excel tables will be created:

1. Subject level information (ID, diagnosis_folder, disgnosis_clinical_file, gender, age, survival, free_survival, nbr_sessions, has_T1w, has_T1wGD, has_T2w, was_FLAIR, has_ADC, has_HandE, has_NameOtherHistologyStains)
2. Session level radiology information (subject ID, session_ID, has_T1w, has_T1wGD, has_T2w, was_FLAIR, has_ADC)
3. File level radiology information (subject ID, session_ID, MR_sequence_type, 2D_or_3D, resolution_x, resolution_y, resolution_x, orientation, scanner)

STEPS
We build the information for each subject top-down, going from the clinical files, the available radiology and histology sessions and the files.
"""

import os
import sys
import csv
import glob
import pathlib
import openpyxl
import numpy as np
import json
import pandas as pd
from copy import deepcopy
import nibabel as nib
import pickle

# local imports
sys.path.append(os.path.join(os.gcwd, "data_utilities"))

from dataset_scraping_utilities import (
    check_pre_or_post_operative_session,
    get_empty_overall_subject_info,
    get_radiology_acquisition_information,
    get_radiology_information_from_volume,
)

# %% DEFINE PATHS
RADIOLOGY_DATASET_PATH = "../CBTN/RADIOLOGY"

CLINICAL_FILE_FROM_PORTAL_PATH = "../CBTN_clinical_data_from_portal.xlsx"
BIOSPECIMEN_FILE_FROM_PORTAL_PATH = "../CBNT_biospecimenData_from_portal.xlsx"

CLINICAL_FILES_PATHS = {
    "ATRT": "../ATRT_clinical_information.xlsx",
    "DIPG": "../DIPG_clinical_information.xlsx",
    "LGG": "../LGG_clinical_information.xlsx",
    "MEDULLO": "../Medullo_clinical_information.xlsx",
}

SAVE_PATH = ""

# %% GET UNIQUE SUBJECT IDs FROM THE RADIOLOGY DATASET FOLDER

tumor_types = {
    "ATRT": "ATRT",
    "DIPG": "DIPG",
    "HGG": "HGG",
    "MEDULLOBLASTOMA": "Medullo",
    "CRANIOPHARYNGIOMA": "Craniopharyngioma",
    "EPENDYMOMA": "Ependymoma",
    "LGG": "LGG",
}

# each element in the subject ID has the ID, tumor_type based on the radiology folder, radiology_folder, histology_folder)
subject_IDs = {}

# through the radiology dataset
for tumor_type, folder_name in tumor_types.items():
    subjects = [
        s.split(os.path.sep)[-2]
        for s in glob.glob(
            os.path.join(RADIOLOGY_DATASET_PATH, folder_name, "SUBJECTS", "*/")
        )
    ]
    # add to the dictionary
    for s in list(dict.fromkeys(subjects)):
        subject_IDs[s] = {
            "tumor_type": tumor_type,
            "radiology_folder": os.path.join(
                RADIOLOGY_DATASET_PATH, folder_name, "SUBJECTS"
            ),
            "histology_folder": None,
        }
    print(f"Tumor Type: {tumor_type} -> {len(subjects)} subjects")
# get the unique values


# %% OPEN CLINICAL FILES (both the ones for each tumor and the ones from the portal)
"""
The files from the portal should cover all the subject IDs available in the radiology folders.
Thus we use these as starting points to gather the information from every subject. The information from the portal
files is integrated with the ones specific to each tumor (if the tumor type and subject are available). 
"""

# open the files (just open)
print("Opening clinical and biospecimens file from Kids First portal...")
clinical_file_from_portal_patients_info = pd.read_excel(
    CLINICAL_FILE_FROM_PORTAL_PATH, sheet_name="Participants", index_col="External Id"
)
clinical_file_from_portal_diagnosis_info = pd.read_excel(
    CLINICAL_FILE_FROM_PORTAL_PATH, sheet_name="Diagnoses", index_col="External Id"
)
biospecimen_file_from_portal = pd.read_excel(
    BIOSPECIMEN_FILE_FROM_PORTAL_PATH,
    sheet_name="Biospecimens",
    index_col="External Id",
)

# filter to only use CBTNs data
clinical_file_from_portal_patients_info = clinical_file_from_portal_patients_info[
    clinical_file_from_portal_patients_info["Study"]
    == "Pediatric Brain Tumor Atlas: CBTTC"
]
clinical_file_from_portal_diagnosis_info = clinical_file_from_portal_diagnosis_info[
    clinical_file_from_portal_diagnosis_info.index.isin(
        clinical_file_from_portal_patients_info.index
    )
]
biospecimen_file_from_portal = biospecimen_file_from_portal[
    biospecimen_file_from_portal.index.isin(
        clinical_file_from_portal_patients_info.index
    )
]
print("Done!")

print("Opening per-tumor type clinical files...")
clinical_file_per_tumor = pd.concat(
    [
        pd.read_excel(f, sheet_name=1, index_col=0)
        for f in CLINICAL_FILES_PATHS.values()
        if os.path.isfile(f)
    ]
)
print("Done!")


# %% SCRAPE CLINICAL FILES (only the subjects that appear in the radiology folders)

# for each subject we save the different information
all_information = {}

"""
For the unique subjects, extract the information that is available from the clinical_file_from_portal (sex, diagnosis, age_at_diagnosis, overall survival)

NB! 
There are subject_IDs which have multiple versions in the Participants list. However, only one version of them has
the information. Thus, one needs to drop all those cases to avoid errors in the code later
"""

# just for printing
aus_len = len(str(len(clinical_file_from_portal_patients_info.index.unique())))

for idx, subject in enumerate(clinical_file_from_portal_patients_info.index.unique()):
    print(
        f"Working on subject {idx+1:0{aus_len}d}/{len(clinical_file_from_portal_patients_info.index.unique())} \r",
        end="",
    )
    if subject in subject_IDs.keys():
        # build entry in the all_information dictionary
        all_information[subject] = get_empty_overall_subject_info()

        # get information from the clinical_file_from_portal_patients_info
        for ikey, ykey in zip(
            ["gender", "overall_survival", "vital_status", "ethnicity"],
            [
                "Gender",
                "Age at the Last Vital Status (Days)",
                "Vital Status",
                "Ethnicity",
            ],
        ):
            try:
                all_information[subject][
                    ikey
                ] = clinical_file_from_portal_patients_info.loc[subject][ykey]
            except:
                continue

        # get information from the clinical_file_from_portal_diagnosis_info: age_at_diagnosis, diagnosis and location
        # this will be refined in a later stage based on the age_at_sample_acquisition_histology which is the
        # one used for the diagnosis and the location and description of the tumor.
        try:
            aus_info = (
                clinical_file_from_portal_diagnosis_info.loc[[subject]][
                    [
                        "Age at Diagnosis (Days)",
                        "Diagnosis (Source Text)",
                        "Tumor Location",
                    ]
                ]
                .dropna(subset=["Age at Diagnosis (Days)"])
                .sort_values(by=["Age at Diagnosis (Days)"])
                .drop_duplicates(
                    subset=["Age at Diagnosis (Days)", "Diagnosis (Source Text)"]
                )
            )

            all_information[subject]["age_at_diagnosis"] = list(
                aus_info["Age at Diagnosis (Days)"]
            )
            all_information[subject]["diagnosis"] = list(
                aus_info["Diagnosis (Source Text)"]
            )
            all_information[subject]["location"] = list(aus_info["Tumor Location"])
        except:
            continue

        # get information from the biospecimen_file_from_portal (this is where the histology information
        # is stored)
        aus_info = (
            biospecimen_file_from_portal.loc[[subject]][
                ["Age at Sample Acquisition", "Tumor Descriptor"]
            ]
            .dropna(subset=["Age at Sample Acquisition"])
            .sort_values(by=["Age at Sample Acquisition"])
            .drop_duplicates(subset=["Age at Sample Acquisition", "Tumor Descriptor"])
        )
        all_information[subject]["age_at_sample_acquisition_histology"] = list(
            aus_info["Age at Sample Acquisition"]
        )

        all_information[subject]["tumor_descriptor"] = list(
            aus_info["Tumor Descriptor"]
        )

        # refine the tumor diagnosis by taking out only the values at age_at_sample_acquisition_histology
        # Here only take the overlapping information (the clinical file and the biospecimen file might have extra
        # information. but only take the overpalling information)
        indexes = [
            all_information[subject]["age_at_diagnosis"].index(i)
            for i in all_information[subject]["age_at_sample_acquisition_histology"]
            if i in all_information[subject]["age_at_diagnosis"]
        ]
        all_information[subject]["diagnosis"] = [
            all_information[subject]["diagnosis"][i] for i in indexes
        ]
        all_information[subject]["location"] = [
            all_information[subject]["location"][i] for i in indexes
        ]

        indexes = [
            all_information[subject]["age_at_sample_acquisition_histology"].index(
                all_information[subject]["age_at_diagnosis"][i]
            )
            for i in indexes
        ]
        all_information[subject]["age_at_sample_acquisition_histology"] = [
            all_information[subject]["age_at_sample_acquisition_histology"][i]
            for i in indexes
        ]
        all_information[subject]["tumor_descriptor"] = [
            all_information[subject]["tumor_descriptor"][i] for i in indexes
        ]

        # from the per-tumor type clinical files get the progression_free_survival and
        # check that the survival age is smaller than the one already available. If not substiute.
        if subject in clinical_file_per_tumor.index:
            all_information[subject][
                "progression_free_survival"
            ] = clinical_file_per_tumor.loc[[subject]]["Progression Free Survival"]

            # get overall survival (these can be many values as well as 'Not Reported')
            overall_survival_from_per_tumor_clinical_file = clinical_file_per_tumor.loc[
                subject
            ]["Overall Survival"]
            if isinstance(overall_survival_from_per_tumor_clinical_file, str):
                overall_survival_from_per_tumor_clinical_file = 0
            else:
                overall_survival_from_per_tumor_clinical_file = np.max(
                    overall_survival_from_per_tumor_clinical_file
                )
            # check overall survival
            if (
                overall_survival_from_per_tumor_clinical_file
                > all_information[subject]["overall_survival"]
            ):
                all_information[subject][
                    "overall_survival"
                ] = clinical_file_per_tumor.loc[subject]["Overall Survival"]

        # if idx == 100:
        #     break

# and finally check that all the files that have been included
missing_info = []
for subject_ID in subject_IDs.keys():
    if subject_ID not in all_information.keys():
        missing_info.append(subject_ID)
if len(missing_info) != 0:
    print(
        f"Missing clinical information for {len(missing_info)} subjects. Adding information..."
    )
    for subject in missing_info:
        all_information[subject] = get_empty_overall_subject_info()


# %% LOOP THROUGH THE UNIQUE Subjects AND SCRAPE THE DATA (from the dataset folders)

list_of_unknowns = []
for subj_idx, (subject, subj_path_info) in enumerate(subject_IDs.items()):
    # create entry in the all_information and add the subject-related information
    tumor_type_from_folder, radiology_folder = (
        subj_path_info["tumor_type"],
        subj_path_info["radiology_folder"],
    )

    all_information[subject]["diagnosis_from_folder"] = (
        tumor_type_from_folder if tumor_type_from_folder else "None"
    )

    # now work on scraping the radiology information if available
    all_information[subject]["nbr_radiology_sessions"] = {
        "pre_op": 0,
        "post_op": 0,
        "unknown": 0,
    }
    if radiology_folder:
        session_iter = glob.glob(
            os.path.join(radiology_folder, subject, "SESSIONS", "*/")
        )
        if len(session_iter) != 0:
            # add space for the radiology sessions
            all_information[subject]["radiology_sessions"] = {}
            # work on every session
            for session_idx, session in enumerate(session_iter):
                # get the session name from the path
                session_name = session.split(os.path.sep)[-2]
                # only process those that have brain in the name
                if not "B_brain" in session_name:
                    continue
                else:
                    # create entry in the all_information dict for this session
                    all_information[subject]["radiology_sessions"][session_name] = {}

                    # get if the session is pre or post operative (this works for the subjects that have clinical information)
                    try:
                        session_status = check_pre_or_post_operative_session(
                            all_information[subject][
                                "age_at_sample_acquisition_histology"
                            ],
                            all_information[subject]["tumor_descriptor"],
                            int(session_name.split("d")[0]),
                        )
                        all_information[subject]["radiology_sessions"][session_name][
                            "pre_post_operation_status"
                        ] = session_status
                        # count session
                        all_information[subject]["nbr_radiology_sessions"][
                            session_status
                        ] += 1
                    except KeyError:
                        all_information[subject]["radiology_sessions"][session_name][
                            "pre_post_operation_status"
                        ] = "unknown"
                        all_information[subject]["nbr_radiology_sessions"][
                            "unknown"
                        ] += 1
                    # work on every acquisition in the session
                    acquisition_iter = glob.glob(
                        os.path.join(session, "ACQUISITIONS", "*/")
                    )
                    for acquisition_idx, acquisition in enumerate(acquisition_iter):
                        # get acquisition name
                        acquisition_name = acquisition.split(os.path.sep)[-2]
                        # do not scrape the acquisitions which are localize
                        target = [
                            "localizer",
                            "loc",
                            "aahscout",
                            "tumble",
                            "reformat",
                            "rotate",
                            "aahead",
                        ]
                        if any([t in acquisition_name.lower() for t in target]):
                            continue
                        else:
                            # open the .json file with all the information about the acquisition
                            # here the file that we want to open is the one that has the same name as the .nii.gz file but with .json extension
                            # get the files with .nii.gz extension
                            niigz_files = glob.glob(
                                os.path.join(acquisition, "FILES", "*.nii.gz")
                            )
                            # there can be multiple .nii.gz files, thus process each independently
                            for niigz_file_idx, niigz_file in enumerate(niigz_files):
                                # get file name
                                file_name = niigz_file.split(os.path.sep)[-1]
                                # get information from the acquisition .json file
                                acquisition_information_file = os.path.join(
                                    acquisition,
                                    "FILES",
                                    file_name.split(".nii.gz")[0] + ".json",
                                )
                                # try:
                                (
                                    MR_sequence,
                                    acquisition_info,
                                ) = get_radiology_acquisition_information(
                                    acquisition_information_file
                                )

                                # get information by opening the volume
                                acquisition_info.update(
                                    get_radiology_information_from_volume(
                                        os.path.join(acquisition, "FILES", file_name)
                                    )
                                )
                                # save the infromation in the all_information
                                if (
                                    MR_sequence
                                    not in all_information[subject][
                                        "radiology_sessions"
                                    ][session_name].keys()
                                ):
                                    all_information[subject]["radiology_sessions"][
                                        session_name
                                    ][MR_sequence] = []
                                all_information[subject]["radiology_sessions"][
                                    session_name
                                ][MR_sequence].append(acquisition_info)

                                # save the unknowns file names
                                if MR_sequence == "UNKNOWN":
                                    list_of_unknowns.append(
                                        f'{subject}, {session_name}, {acquisition_info["file_name"]}'
                                    )
                                # except:
                                # continue
    # print status
    print(
        f"Processing subject {subj_idx:04d}/{len(subject_IDs)}\r",
        end="",
    )

    # if len(list_of_unknowns) > 100:
    #     break

# %% SAVE WITH PICKLE
for s, i in all_information.items():
    for ss, j in i.items():
        if isinstance(j, pd.Series):
            all_information[s][ss] = list(j)

pathlib.Path(
    os.path.join(SAVE_PATH, "summary_scraped_radiology_dataset.pkl")
).write_bytes(pickle.dumps(all_information))

# %% RE_ORGANIZE TO HAVE ONE DICT ITEMS PER SESSION

MR_sequences = [
    "T1W",
    "T1W_SE",
    "T1W_FL",
    "T1W_MPRAGE",
    "T1W_GD",
    "T1W_SE_GD",
    "T1W_FL_GD",
    "T1W_MPRAGE_GD",
    "T2W",
    "T2W_TRIM",
    "T2W_FLAIR",
    "DIFFUSION",
    "TRACE",
    "ADC",
    "FA",
    "EXP",
    "UNKNOWN",
    "PERFUSION",
    "SWI",
    "MAG",
    "PHE",
    "ASL",
    "SWAN",
    "CBF",
    "ANGIO",
    "FLAIR",
]

all_information_session_wise = []
entries_count = 0
for idx, (subj_id, subj_data) in enumerate(all_information.items()):
    if "radiology_sessions" in subj_data.keys():
        for session_id, session_data in subj_data["radiology_sessions"].items():
            # gather session level information (diagnosis, age at diagnosis and tumor location)
            """
            Here there are three major possibilities:
            1 - the age at histology diagnosis is not available
                ¤ set the diagnosis, age_at_diagnosis, location and tumor_descriptor to AGE_AT_HISTOLOGY_DIAGNOSIS_NOT_AVAILABLE
            2 - there is only one age_at_sample_acquisition_histology
                ¤ set diagnosis, age_at_diagnosis, location and tumor_descriptor as the one from this age
            3 - there are multiple age_at_sample_acquisition_histology
                ¤ get the closest age_at_sample_acquisition_histology compared to the age_at_sample_acquisition
                and use the information from that age_at_sample_acquisition_histology
            """
            age_at_sample_acquisition = float(session_id.split("d")[0])

            if ("age_at_sample_acquisition_histology" not in subj_data.keys()) or (
                not subj_data["age_at_sample_acquisition_histology"]
            ):
                age_at_diagnosis = "AGE_AT_HISTOLOGY_DIAGNOSIS_NOT_AVAILABLE"
                diagnosis = "AGE_AT_HISTOLOGY_DIAGNOSIS_NOT_AVAILABLE"
                tumor_location = "AGE_AT_HISTOLOGY_DIAGNOSIS_NOT_AVAILABLE"
                tumor_descriptor = "AGE_AT_HISTOLOGY_DIAGNOSIS_NOT_AVAILABLE"
            else:
                if len(subj_data["age_at_sample_acquisition_histology"]) == 1:
                    _index = 0
                else:
                    # there are multiple age_at_sample_acquisition_histology.
                    # Use teh information from the closes one
                    aus = (
                        np.array(subj_data["age_at_sample_acquisition_histology"])
                        - age_at_sample_acquisition
                    ) ** 2
                    _index = np.argsort(aus)[0]

                # save infromation
                age_at_diagnosis = subj_data["age_at_sample_acquisition_histology"][
                    _index
                ]
                diagnosis = subj_data["diagnosis"][_index]
                tumor_location = subj_data["location"][_index]
                tumor_descriptor = subj_data["tumor_descriptor"][_index]

            # copy the MR modality information (count the number of different modalities)
            for mr_modality, mr_modality_data in session_data.items():
                if mr_modality != "pre_post_operation_status":
                    for image_data in mr_modality_data:
                        # create row entry
                        entry_row = [
                            subj_id,
                            subj_data["gender"],
                            subj_data["ethnicity"],
                            subj_data["overall_survival"],
                            session_id,
                            "RAD",
                            session_data["pre_post_operation_status"],
                            diagnosis,
                            tumor_descriptor,
                            age_at_diagnosis,
                            age_at_sample_acquisition,
                            tumor_location,
                            mr_modality,
                            image_data["MagneticFieldStrength"],
                            "_".join(
                                [
                                    str(image_data["Manufacturer"]),
                                    str(image_data["ManufacturersModelName"]),
                                ]
                            ),
                            image_data["dim_x"],
                            image_data["dim_y"],
                            image_data["dim_z"],
                            image_data["pixels_x"],
                            image_data["pixels_y"],
                            image_data["pixels_z"],
                            image_data["file_name"],
                        ]

                        # finally save information
                        all_information_session_wise.append(entry_row)
                        entries_count += 1
            # print status
            print(
                f"Processing subject {idx+0:04d}/{len(all_information)}\r",
                end="",
            )


# %% EXPORT TO excel
columns = [
    "subjectID",
    "gender",
    "ethnicity",
    "survival",
    "session_name",
    "type_of_session",
    "session_status",
    "diagnosis",
    "tumor_descriptor",
    "age_at_diagnosis",
    "age_at_sample_acquisition",
    "tumor_location",
    "image_type",
    "magnification",
    "scanner",
    "dimension_x",
    "dimension_y",
    "dimension_z",
    "pixels_x",
    "pixels_y",
    "pixels_z",
    "file_name",
]

df = pd.DataFrame(all_information_session_wise, columns=columns)

df.to_csv(os.path.join(SAVE_PATH, "radiology_per_file_scraped_data.csv"), index=False)
