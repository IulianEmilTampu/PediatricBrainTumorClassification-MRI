# %%
"""
Script that given the data_split_information and the clinical data information, counts the subjects (slices) used
for all the modalities, along with the number of male, female, NA and their age. 
"""

import os
import pandas as pd
import numpy as np

# %% SOURCE FILES

CLINICAL_INFORMATION = "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/count_used_subjects_slices/CBTN_clinical_data_from_portal.xlsx"
PER_MODALITY_SPLIT_INFORMATION = {
    "ADC": "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/count_used_subjects_slices/ADC_data_split_information.csv",
    "T1": "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/count_used_subjects_slices/T1_data_split_information.csv",
    "T2": "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/count_used_subjects_slices/T2_data_split_information.csv",
}

# load the clinical file
clinical_information = pd.read_excel(CLINICAL_INFORMATION, sheet_name=0)

# %% FOR EACH MODALITY PLOT SUMMARY

summary_df = []
for modality_name, split_information_file in PER_MODALITY_SPLIT_INFORMATION.items():
    # load the split information
    split_information = pd.read_csv(split_information_file)
    # NOTE! Very important to order based on subject ID since unique and group by do not do that, thus the values are not aligned.
    split_information = split_information.sort_values(by=["subject_IDs"])
    # get the unique subject IDs
    unique_subjectIDs = list(pd.unique(split_information.subject_IDs))
    # get slice count
    slices_per_subject = list(
        split_information.groupby(["subject_IDs"]).count()["file_path"]
    )
    # get diagnosis
    diagnosis = list(
        split_information.groupby(["subject_IDs"]).apply(
            lambda x: list(pd.unique(x.target))[0]
        )
    )
    # for each of them, get the gender from the clinical file
    gender = [
        list(
            pd.unique(
                clinical_information.loc[clinical_information["External Id"] == s][
                    "Gender"
                ]
            )
        )
        for s in unique_subjectIDs
    ]
    for idx, g in enumerate(gender):
        if g:
            gender[idx] = g[0]
        else:
            gender[idx] = "NaN"
    # get age in days (this is the smales age in days among the age of this subject)
    age = list(
        split_information.groupby(["subject_IDs"]).apply(lambda x: min(x.age_in_days))
    )

    # all the information now is available. Build new DF for counting
    df = pd.DataFrame(
        data=[
            unique_subjectIDs,
            gender,
            age,
            diagnosis,
            slices_per_subject,
            [modality_name] * len(unique_subjectIDs),
        ]
    ).transpose()
    df.columns = [
        "subject_IDs",
        "gender",
        "age_in_days",
        "diagnosis",
        "nbr_slices",
        "mr_sequence",
    ]

    # save
    summary_df.append(df)

summary_df = pd.concat(summary_df)

# %% COUNT

# DIAGNOSTIC LEVEL FIRST
for diagnosis in ["ASTROCYTOMA", "EPENDYMOMA", "MEDULLOBLASTOMA"]:
    # SUBJECT LEVEL INFORMATION
    df = summary_df.loc[summary_df.diagnosis == diagnosis]
    indent = 4
    nbr_subjects = len(pd.unique(df.subject_IDs))
    gender_count = list(df.groupby(["gender"]).nunique()["subject_IDs"])
    males, females, NAs = (
        gender_count[1],
        gender_count[0],
        gender_count[2] if len(gender_count) == 3 else 0,
    )
    age_median, age_min, age_max, age_mean, age_std = (
        np.median(df.age_in_days) / 365,
        np.min(df.age_in_days) / 365,
        np.max(df.age_in_days) / 365,
        np.mean(df.age_in_days) / 365,
        np.std(df.age_in_days) / 365,
    )

    print(f"{diagnosis}")
    indent = 4
    print(
        f"{' '*indent:s}{nbr_subjects} subjects. (M/F/NA) {'/'.join([str(v) for v in [males, females, NAs]])}"
    )
    print(
        f"{' '*indent:s}Age: (median [min, max]): {age_median:0.2f} [{age_min:0.2f}, {age_max:0.2f}]"
    )
    print(f"{' '*indent:s}Age: (mean +- std): {age_mean:0.2f} +- {age_std:0.2f}")

    for mr_sequence in ["T2", "T1", "ADC"]:
        indent = 4
        print(f"{' '*indent:s}{mr_sequence}")
        indent = 8
        df_seq = df.loc[df.mr_sequence == mr_sequence]
        nbr_subjects_sequence = len(pd.unique(df_seq.subject_IDs))
        nbr_slices = np.sum(df_seq.nbr_slices)
        print(f"{' '*indent:s}{nbr_subjects_sequence} subjects")
        print(f"{' '*indent:s}{nbr_slices} slices")

    print("\n")


# %%
print("\nTOTALS")

# TOTALS
nbr_subjects = len(pd.unique(summary_df.subject_IDs))
gender_count = list(summary_df.groupby(["gender"]).nunique()["subject_IDs"])
males, females, NAs = (
    gender_count[1],
    gender_count[0],
    gender_count[2] if len(gender_count) == 3 else 0,
)
age_median, age_min, age_max, age_mean, age_std = (
    np.median(summary_df.age_in_days) / 365,
    np.min(summary_df.age_in_days) / 365,
    np.max(summary_df.age_in_days) / 365,
    np.mean(summary_df.age_in_days) / 365,
    np.std(summary_df.age_in_days) / 365,
)

print(
    f"{' '*indent:s}{nbr_subjects} subjects. (M/F/NA) {'/'.join([str(v) for v in [males, females, NAs]])}"
)
print(
    f"{' '*indent:s}Age: (median [min, max]): {age_median:0.2f} [{age_min:0.2f}, {age_max:0.2f}]"
)
print(f"{' '*indent:s}Age: (mean +- std): {age_mean:0.2f} +- {age_std:0.2f}")

print("\n")
for mr_sequence in ["T2", "T1", "ADC"]:
    indent = 4
    print(f"{' '*indent:s}{mr_sequence}")
    indent = 8
    df_seq = summary_df.loc[summary_df.mr_sequence == mr_sequence]
    nbr_subjects_sequence = len(pd.unique(df_seq.subject_IDs))
    nbr_slices = np.sum(df_seq.nbr_slices)
    print(f"{' '*indent:s}{nbr_subjects_sequence} subjects")
    print(f"{' '*indent:s}{nbr_slices} slices")


# %%
