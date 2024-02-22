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
PATH_TO_FILTER_PREPROCESSING_FILE = "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/count_used_subjects_slices/checked_files_min_slice_number_50_maxres_1mm_20230116.xlsx"
PATH_TO_PREVIOUS_EXPERIMENTS_EXCEL_FILES = "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/count_used_subjects_slices/previous_analysis_excel_files"

# load the clinical file
clinical_information = pd.read_excel(CLINICAL_INFORMATION, sheet_name=0)

# load filter_preprocessing_file
filter_preprocessing_file = pd.read_excel(PATH_TO_FILTER_PREPROCESSING_FILE)

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


# %% PRINT TOTALS
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


# %% CHECK WITH THE FILTER EXCEL FILE TO SEE IF SOME WERE NOT PRE-PROCESSED
unique_subjects_passed_filtering = pd.unique(
    filter_preprocessing_file.loc[
        filter_preprocessing_file.final_check == "passed"
    ].subject_ID
)
unique_subjects_pre_processed = pd.unique(summary_df.subject_IDs)

subjects_not_preprocessed = [
    v
    for v in unique_subjects_passed_filtering
    if v not in unique_subjects_pre_processed
]
subjects_preprocessed_but_out_of_filter = [
    v
    for v in unique_subjects_pre_processed
    if v not in unique_subjects_passed_filtering
]

print(
    f"Out of the {len(unique_subjects_passed_filtering)}, {len(subjects_not_preprocessed)} were not preprocessed."
)
print([print(i) for i in subjects_not_preprocessed])


print(
    f"\nSubjects not in the filter that are part of the dataset {len(subjects_preprocessed_but_out_of_filter)}."
)
print([print(i) for i in subjects_preprocessed_but_out_of_filter])

# %% DO THIS COUNT PER MODALITY
filter_preprocessing_file = filter_preprocessing_file.loc[
    filter_preprocessing_file.final_check == "passed"
]
# for every modality, count the subjects that are similar and different
for mr_sequence in pd.unique(summary_df.mr_sequence):
    # get unique subjects from the current analysis
    current_analysis_subjects = pd.unique(
        summary_df.loc[summary_df.mr_sequence == mr_sequence].subject_IDs
    )
    # get the unique subjects from the filter file
    filter_file_subjects = pd.unique(
        filter_preprocessing_file.loc[
            filter_preprocessing_file.actual_modality == mr_sequence
        ].subject_ID
    )
    # subjects in the current analysis but not in the previous
    subjects_in_current_but_not_in_previous = [
        s for s in current_analysis_subjects if s not in filter_file_subjects
    ]
    # and the opposite
    subjects_in_previous_but_not_in_current = [
        s for s in filter_file_subjects if s not in current_analysis_subjects
    ]
    # print findings
    print(
        f"({mr_sequence.upper()}): Subjects in the filter file      : {len(filter_file_subjects)}"
    )
    print(
        f"({mr_sequence.upper()}): Subjects in the current  analysis: {len(current_analysis_subjects)}"
    )
    print(
        f"({mr_sequence.upper()}): Subjects in the filter file but not in current analysis: {len(subjects_in_previous_but_not_in_current)}"
    )
    print(
        f"({mr_sequence.upper()}): Subjects in the current but not in the filter file: {len(subjects_in_current_but_not_in_previous)}"
    )


# %% CHECK WHICH MODALITY THE SUBJECTS NOT PREPROCESSED BELONG TO
for s in subjects_not_preprocessed:
    print(
        pd.unique(
            filter_preprocessing_file.loc[
                (filter_preprocessing_file.subject_ID == s)
                & (filter_preprocessing_file.final_check == "passed")
            ].actual_modality
        )
    )
# %% CHECK WHICH SUBJECT COMPARED TO THE PREVIOUS ANALYSIS ARE INCLUDED/EXCLUDED

PATH_TO_PREVIOUS_ANALYSIS_FILES = {
    "ADC": "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/count_used_subjects_slices/previous_analysis_excel_files/adc_all_files.xlsx",
    "T1": "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/count_used_subjects_slices/previous_analysis_excel_files/t1c_all_files.xlsx",
    "T2": "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/count_used_subjects_slices/previous_analysis_excel_files/t2_all_files.xlsx",
}

# load files
df_previous_analysis = []
for mr_sequence, p in PATH_TO_PREVIOUS_ANALYSIS_FILES.items():
    df = pd.read_excel(p)
    df["mr_sequence"] = mr_sequence
    df_previous_analysis.append(df)

df_previous_analysis = pd.concat(df_previous_analysis)

# for every modality, count the subjects that are similar and different
for mr_sequence in pd.unique(summary_df.mr_sequence):
    # get unique subjects from the current analysis
    current_analysis_subjects = pd.unique(
        summary_df.loc[summary_df.mr_sequence == mr_sequence].subject_IDs
    )
    # get the unique subjects from the previous analysis
    previous_analysis_subjects = pd.unique(
        df_previous_analysis.loc[
            df_previous_analysis.mr_sequence == mr_sequence
        ].subject_ID
    )
    # subjects in the current analysis but not in the previous
    subjects_in_current_but_not_in_previous = [
        s for s in current_analysis_subjects if s not in previous_analysis_subjects
    ]
    # and the opposite
    subjects_in_previous_but_not_in_current = [
        s for s in previous_analysis_subjects if s not in current_analysis_subjects
    ]
    # print findings
    print(
        f"({mr_sequence.upper()}): Subjects in the previous analysis: {len(previous_analysis_subjects)}"
    )
    print(
        f"({mr_sequence.upper()}): Subjects in the current  analysis: {len(current_analysis_subjects)}"
    )
    print(
        f"({mr_sequence.upper()}): Subjects in the previous but not in current analysis: {len(subjects_in_previous_but_not_in_current)}"
    )
    print(
        f"({mr_sequence.upper()}): Subjects in the current but not in previous analysis: {len(subjects_in_current_but_not_in_previous)}"
    )

# %% SAVE FOR PLOTTING
"""
Save also df for plotting (as in CBTN_v2 project)
subjectID | gender | ethnicity | survival | session_name | type_of_session | session_status | diagnosis | age_at_diagnosis | image_type 
"""
for_plotting_df = []

for modality_name, split_information_file in PER_MODALITY_SPLIT_INFORMATION.items():
    split_information = pd.read_csv(split_information_file)
    # session_name from the file_path
    split_information["session_name"] = split_information.apply(
        lambda x: pathlib.Path(x.file_path).parts[-1].split("___")[3], axis=1
    )
    # add gender information
    split_information["gender"] = split_information.apply(
        lambda x: summary_df.loc[summary_df.subject_IDs == x.subject_IDs].gender.values[
            0
        ],
        axis=1,
    )
    # add the session type information
    split_information["session_type"] = ["RAD"] * len(split_information)
    # add session_status
    split_information["session_status"] = ["pre_op"] * len(split_information)
    # add image_type
    split_information["image_type"] = [modality_name] * len(split_information)
    # rename columns
    split_information = split_information.rename(
        columns={
            "subject_IDs": "subjectID",
            "target": "diagnosis",
            "age_in_days": "age_at_diagnosis",
        }
    )
    # drop the information that is not needed
    split_information = split_information.drop(
        columns=[
            "file_path",
            "tumor_relative_position",
            "fold_1",
            "fold_2",
            "fold_3",
            "fold_4",
            "fold_5",
            "age_normalized",
        ]
    )
    # save
    for_plotting_df.append(split_information)

# %%
# save to file
for_plotting_df = pd.concat(for_plotting_df)
for_plotting_df.to_csv(
    os.path.join(os.path.dirname(CLINICAL_INFORMATION), "summary_for_plotting.csv")
)
