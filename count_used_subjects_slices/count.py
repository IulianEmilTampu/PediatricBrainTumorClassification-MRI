# %%
"""
Script that given the data_split_information and the clinical data information, counts the subjects (slices) used
for all the modalities, along with the number of male, female, NA and their age. 
"""

import os
import pandas as pd

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
        data=[unique_subjectIDs, gender, age, diagnosis, slices_per_subject, [modality_name]*len(unique_subjectIDs)]
    ).transpose()
    df.columns = ["subject_IDs", "gender", "age_in_days", "diagnosis", "nbr_slices", 'mr_sequence']

    # save
    summary_df.append(df)

summary_df = pd.concat(summary_df)

# %% COUNT 
# DIAGNOSTIC LEVEL FIRST 
for diagnosis in list(pd.unique(summary_df.diagnosis)):
    print(f'Diagnosis: {diagnosis}')
    # SUBJECT LEVEL INFORMATION
    df = summary_df.loc[summary_df.diagnosis==diagnosis]
    indent = 4
    print(f'{" "*indent:s}{len(pd.unique(df.subject_IDs))} unique subjects. Age (in years) {}')
    print(f'{" "*indent:s}F/M/NA subjects {"/".join([str(v) for v in list(df.groupby(["gender"]).count()["subject_IDs"])])}')
    print(f'{" "*indent:s}F/M/NA slices {"/".join([str(v) for v in list(df.groupby(["gender"]).count()["nbr_slices"])])}')
    # age 
    print(f'{" "*indent:s}{len(pd.unique(df.subject_IDs))} unique subjects.')


    # # start counting 
    # print(f'Modality {modality_name}')
    # # count unique subjects, number of male, female and NaN
    # indent = 4
    # print(f'{" "*indent:s}{len(unique_subjectIDs)} unique subjects.')
    # print(f'{" "*indent:s}F/M/NA subjects {"/".join([str(v) for v in list(df.groupby(["gender"]).count()["subject_IDs"])])}')
    # print(f'{" "*indent:s}F/M/NA slices {"/".join([str(v) for v in list(df.groupby(["gender"]).count()["nbr_slices"])])}')
# %%
