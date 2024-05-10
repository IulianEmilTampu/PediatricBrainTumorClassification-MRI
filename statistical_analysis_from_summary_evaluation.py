# %%

""" 
Script that uses the summary_evaluation_aggregated.csv file to statistically compare:
1 - if addition of the age information changes model performance (keeping constant: MR sequence, model version and pre-training strategy)
2 - ResNet50 - vs - ViT_b_16 (keeping constant: MR sequence, pre-training strategy and input (with or without age))
3 - T1 - vs - T2 - vs - ADC: the best model (chosen between model version, pre-training strategy and input). (Bonferoni correction needed.)
"""

import os
import glob
import time
import csv
import json
import pathlib
import platform
import numpy as np
import pandas as pd
import seaborn as sns
from copy import deepcopy
from itertools import cycle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
from scipy import stats
import itertools

# %% IMPORT SUMMARY FILE
SUMMARY_FILE_PATH = "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/evaluation_results/Evaluation_20240212/summary_evaluation_aggregated.csv"
SAVE_PATH = pathlib.Path(
    os.path.join(os.path.dirname(SUMMARY_FILE_PATH), "Summary_statistical_analysis")
)
SAVE_PATH.mkdir(parents=True, exist_ok=True)
ORIGINAL_DF = pd.read_csv(SUMMARY_FILE_PATH)

# SET to look at
EVALUATION_SET = "test"
ORIGINAL_DF = ORIGINAL_DF.loc[ORIGINAL_DF.evaluation_set == EVALUATION_SET]
# number of classes
NBR_CLASSES = 3
ORIGINAL_DF = ORIGINAL_DF.loc[ORIGINAL_DF.nbr_classes == NBR_CLASSES]
# use model with 0.5 fine tuning
ORIGINAL_DF = ORIGINAL_DF.loc[ORIGINAL_DF.fine_tuning == 0.5]
# use plot results for a 3-class classification problem
ORIGINAL_DF = ORIGINAL_DF.loc[ORIGINAL_DF.nbr_classes == 3]


# add a summary pre-training string which is used for later grouping the plots
# pretraining==False -> ImageNet
# pretraining==True $ pretraining_dataset == tcga -> SimCLR_TCGA
# pretraining==True $ pretraining_dataset == cbtn -> SimCLR_CBTN
def make_pretraining_type_string(x):
    if not x.pretraining:
        return "ImageNet"
    else:
        return f"SimCLR_{str(x.pretraining_dataset).upper()}"


ORIGINAL_DF["pretraining_type_str"] = ORIGINAL_DF.apply(
    lambda x: make_pretraining_type_string(x), axis=1
)

# make one dataframe for the slice-level plotting and one for the subject-level plotting
# get subject level performance
FILTER = [f"subject_ensemble_pred_fold_{i+1}" for i in range(10)]
ORIGINAL_DF_SUBJECT_LEVEL = ORIGINAL_DF[ORIGINAL_DF["performance_over"].isin(FILTER)]
ORIGINAL_DF_SUBJECT_LEVEL_ENSEMBLE = ORIGINAL_DF[
    ORIGINAL_DF["performance_over"] == "overall_subject_ensemble"
]

# get slice level performance
FILTER = [f"pred_fold_{i+1}" for i in range(10)]
ORIGINAL_DF_SLICE_LEVEL = ORIGINAL_DF[ORIGINAL_DF["performance_over"].isin(FILTER)]
ORIGINAL_DF_SLICE_LEVEL_ENSEMBLE = ORIGINAL_DF[
    ORIGINAL_DF["performance_over"] == "per_slice_ensemble"
]


# %% AGE vs NO AGE
# ON THE SAME MR SEQUENCE, MODEL VERSION, PRE-TRAINING, COMPARE THE PERFORMANCE WHEN USING OR NOT THE SUBJECT AGE.
# THIS IS A SUBJECT WISE COMPARISON BETWEEN TWO MODELS ON THE SAME DATA. HERE WE WILL USE THE NON-PARAMETRIC
# WILCOXON SIGNED-RAKED TEST BETWEEN THE THE POPULATION OF SUBJECT-WISE PERFORMANCE FOR EACH FOLD (50 IN TOTAL
# GIVEN 10 TIMES REPEATED 5 FOLD CROSS VALIDATION) WHEN NOT USING SUBJECT AGE AND WHEN USING SUBJECT AGE.

# Here for each mr sequence, model type and for each pretraining configuration, select the two populations
# (without and with age metrics) and perform the comparison and print the results.

unique_things_to_compare = pd.unique(ORIGINAL_DF_SLICE_LEVEL.use_age)
list_of_comparisons = list(itertools.combinations(unique_things_to_compare, 2))
significance_thr = 0.05
bonferroni_corrected_significance_thr = significance_thr / len(list_of_comparisons)

# thinks to keep constant
mr_sequences = ["T1", "T2", "ADC"]
model_versions = ["ResNet50", "ViT_b_16"]
pretraining_versions = ["ImageNet", "SimCLR_TCGA", "SimCLR_CBTN"]

# metrics to test
metric_and_text = {
    "overall_precision": {"metric_name": "Precision [0,1]", "metric_range": [0, 1]},
    "overall_recall": {"metric_name": "Recall [0,1]", "metric_range": [0, 1]},
    "overall_accuracy": {"metric_name": "Accuracy [0,1]", "metric_range": [0, 1]},
    "overall_f1-score": {"metric_name": "F1-score [0,1]", "metric_range": [0, 1]},
    "overall_auc": {"metric_name": "AUC [0,1]", "metric_range": [0, 1]},
    "matthews_correlation_coefficient": {
        "metric_name": "Matthews correlation coefficient [-1,1]",
        "metric_range": [0, 1],
    },
}

# things for string formatting
max_len_metric = np.max([len(v) for v in metric_and_text.keys()])
max_len_mr_sequence = np.max([len(v) for v in mr_sequences])
max_len_model_version = np.max([len(v) for v in model_versions])
max_len_pre_t_version = np.max([len(v) for v in pretraining_versions])
level_indent = 2

# perform analysis and save to file
significant_test_analysis_summary = []
significant_test_analysis_summary_of_the_summary = {
    "significant": [],
    "non_significant": [],
}
significant_test_analysis_summary.append(
    f"Subject-wise statistical analysis of age-vs-noAge."
)
significant_test_analysis_summary.append(
    f"Using a significance threshold of {significance_thr:0.4f} adjusted using Bonferoni correction for multiple comparisons ({len(list_of_comparisons)} comparisons -> {bonferroni_corrected_significance_thr:0.4f} corrected threshold)."
)

for metric, metric_aus in metric_and_text.items():
    print_level = 0
    string = f"{' '*print_level*level_indent:s}Metric: {metric:{max_len_metric}s}"
    significant_test_analysis_summary.append(string)
    for mrs in mr_sequences:
        print_level = 1
        string = (
            f"{' '*print_level*level_indent:s}MR sequence: {mrs:{max_len_mr_sequence}s}"
        )
        significant_test_analysis_summary.append(string)
        print_level = 2
        for mv in model_versions:
            print_level = 3
            string = f"{' '*print_level*level_indent:s}Model version: {mv:{max_len_model_version}s}"
            significant_test_analysis_summary.append(string)
            for ptv in pretraining_versions:
                print_level = 4
                string = f"{' '*print_level*level_indent:s}Pre-training version: {ptv:{max_len_pre_t_version}s}"
                significant_test_analysis_summary.append(string)
                print_level = 5

                # get the two populations
                population_1 = list(
                    ORIGINAL_DF_SUBJECT_LEVEL.loc[
                        (ORIGINAL_DF_SUBJECT_LEVEL.use_age == False)
                        & (ORIGINAL_DF_SUBJECT_LEVEL.mr_sequence == mrs)
                        & (ORIGINAL_DF_SUBJECT_LEVEL.model_version == mv)
                        & (ORIGINAL_DF_SUBJECT_LEVEL.pretraining_type_str == ptv)
                    ][metric]
                )
                population_2 = list(
                    ORIGINAL_DF_SUBJECT_LEVEL.loc[
                        (ORIGINAL_DF_SUBJECT_LEVEL.use_age == True)
                        & (ORIGINAL_DF_SUBJECT_LEVEL.mr_sequence == mrs)
                        & (ORIGINAL_DF_SUBJECT_LEVEL.model_version == mv)
                        & (ORIGINAL_DF_SUBJECT_LEVEL.pretraining_type_str == ptv)
                    ][metric]
                )

                if all(
                    [
                        len(population_1) == len(population_2),
                        all([len(p) != 0 for p in [population_1, population_2]]),
                    ]
                ):
                    # perform test
                    statistical_test = stats.wilcoxon(
                        population_1, population_2, alternative="two-sided"
                    )

                    # print test results
                    string = f"{' '*print_level*level_indent:s}Population 1 (age==False) [mean±std]: {len(population_1)} samples, {np.mean(population_1):0.4f} ± {np.std(population_1):0.4f}"
                    significant_test_analysis_summary.append(string)
                    string = f"{' '*print_level*level_indent:s}Population 2 (age==True) [mean±std]: {len(population_2)} samples, {np.mean(population_2):0.4f} ± {np.std(population_2):0.4f}"
                    significant_test_analysis_summary.append(string)
                    string = f"{' '*print_level*level_indent:s}p-value: {statistical_test[-1]:0.8f} ({'SIGNIFICANT' if statistical_test[-1] <= bonferroni_corrected_significance_thr else 'NOT SIGNIFICANT'})"
                    significant_test_analysis_summary.append(string)
                    # save to the summary of the summary
                    if statistical_test[-1] <= bonferroni_corrected_significance_thr:
                        significant_test_analysis_summary_of_the_summary[
                            "significant"
                        ].append(f"{metric}, {mrs}, {mv}, {ptv}")
                    else:
                        significant_test_analysis_summary_of_the_summary[
                            "non_significant"
                        ].append(f"{metric}, {mrs}, {mv}, {ptv}")
                else:
                    string = f"{' '*print_level*level_indent:s}Skipping statistical test since len populaiton_1=={len(population_1)} and len populaiton_2=={len(population_2)}"
                    significant_test_analysis_summary.append(string)
        significant_test_analysis_summary.append("\n")
    significant_test_analysis_summary.append("\n")


# save also the summary of the summary
for s, s_values in significant_test_analysis_summary_of_the_summary.items():
    string = f"Tests that resulted {s.title()} ({len(s_values)} out of {np.sum([len(v) for v in significant_test_analysis_summary_of_the_summary.values()])})"
    significant_test_analysis_summary.append(string)
    for v in s_values:
        string = v
        significant_test_analysis_summary.append(string)
    string = "\n"
    significant_test_analysis_summary.append(string)

# print summary
for s in significant_test_analysis_summary:
    print(s)

# save to file
summary_file = os.path.join(SAVE_PATH, f"Statistical_analysis_age_vs_noAge.txt")
with open(summary_file, "w") as f:
    for line in significant_test_analysis_summary:
        f.write(f"{line}\n")


# %% PRE-TRAINING COMPARISON
# ON THE SAME MR SEQUENCE, INPUT (WITH OR WITHOUT AGE) and MODEL VERSION COMPARE THE PERFORMANCE WHEN USING DIFFERENT TYPES OF PRE-TRAINING STRATEGIES.
# THIS IS A SUBJECT WISE COMPARISON BETWEEN TWO MODELS ON THE SAME DATA. HERE WE WILL USE THE NON-PARAMETRIC
# WILCOXON SIGNED-RAKED TEST BETWEEN THE THE POPULATION OF SUBJECT-WISE PERFORMANCE FOR EACH FOLD (50 IN TOTAL
# GIVEN 10 TIMES REPEATED 5 FOLD CROSS VALIDATION) WHEN FINE TUNING FROM IMAGENET, SIMCLR_TCGA, SIMCLR_TCGA.

# Here for each mr sequence, pretraining configuration, input configuration select the two populations
# (ResNet50 and ViT_16_b) and perform the comparison and print the results.

unique_things_to_compare = pd.unique(ORIGINAL_DF_SUBJECT_LEVEL.pretraining_type_str)
list_of_comparisons = list(itertools.combinations(unique_things_to_compare, 2))
significance_thr = 0.05
bonferroni_corrected_significance_thr = significance_thr / len(list_of_comparisons)

# thinks to keep constant
mr_sequences = ["T1", "T2", "ADC"]
use_ages = [False, True]
model_versions = ["ResNet50", "ViT_b_16"]
pretraining_versions = ["ImageNet", "SimCLR_TCGA", "SimCLR_CBTN"]

# metrics to test
metric_and_text = {
    "overall_precision": {"metric_name": "Precision [0,1]", "metric_range": [0, 1]},
    "overall_recall": {"metric_name": "Recall [0,1]", "metric_range": [0, 1]},
    "overall_accuracy": {"metric_name": "Accuracy [0,1]", "metric_range": [0, 1]},
    "overall_f1-score": {"metric_name": "F1-score [0,1]", "metric_range": [0, 1]},
    "overall_auc": {"metric_name": "AUC [0,1]", "metric_range": [0, 1]},
    "matthews_correlation_coefficient": {
        "metric_name": "Matthews correlation coefficient [-1,1]",
        "metric_range": [0, 1],
    },
}

# things for string formatting
max_len_metric = np.max([len(v) for v in metric_and_text.keys()])
max_len_mr_sequence = np.max([len(v) for v in mr_sequences])
max_len_model_version = np.max([len(v) for v in model_versions])
max_len_pre_t_version = np.max([len(v) for v in pretraining_versions])
max_len_use_age = np.max([len(str(v)) for v in use_ages])
level_indent = 2

# perform analysis and save to file
significant_test_analysis_summary = []
significant_test_analysis_summary_of_the_summary = {
    "significant": [],
    "non_significant": [],
}
significant_test_analysis_summary.append(
    f"Subject-wise statistical analysis of ResNet50 vs ViT_b_16"
)
significant_test_analysis_summary.append(
    f"Using a significance threshold of {significance_thr:0.4f} adjusted using Bonferoni correction for multiple comparisons ({len(list_of_comparisons)} comparisons -> {bonferroni_corrected_significance_thr:0.4f} corrected threshold)."
)

for metric, metric_aus in metric_and_text.items():
    print_level = 0
    string = f"{' '*print_level*level_indent:s}Metric: {metric:{max_len_metric}s}"
    significant_test_analysis_summary.append(string)
    for mrs in mr_sequences:
        print_level = 1
        string = (
            f"{' '*print_level*level_indent:s}MR sequence: {mrs:{max_len_mr_sequence}s}"
        )
        significant_test_analysis_summary.append(string)
        for mv in model_versions:
            print_level = 3
            string = f"{' '*print_level*level_indent:s}Model version: {mv:{max_len_model_version}s}"
            significant_test_analysis_summary.append(string)
            for use_age in use_ages:
                print_level = 2
                string = f"{' '*print_level*level_indent:s}Use age: {str(use_age):{max_len_use_age}s}"
                significant_test_analysis_summary.append(string)
                for ptv_comparison in list_of_comparisons:
                    print_level = 3
                    string = f"{' '*print_level*level_indent:s}Comparison: {'-vs-'.join(ptv_comparison):40s}"
                    significant_test_analysis_summary.append(string)
                    print_level = 4

                    # get the two populations
                    population_1 = list(
                        ORIGINAL_DF_SUBJECT_LEVEL.loc[
                            (ORIGINAL_DF_SUBJECT_LEVEL.use_age == use_age)
                            & (ORIGINAL_DF_SUBJECT_LEVEL.mr_sequence == mrs)
                            & (ORIGINAL_DF_SUBJECT_LEVEL.model_version == mv)
                            & (
                                ORIGINAL_DF_SUBJECT_LEVEL.pretraining_type_str
                                == ptv_comparison[0]
                            )
                        ][metric]
                    )
                    population_2 = list(
                        ORIGINAL_DF_SUBJECT_LEVEL.loc[
                            (ORIGINAL_DF_SUBJECT_LEVEL.use_age == use_age)
                            & (ORIGINAL_DF_SUBJECT_LEVEL.mr_sequence == mrs)
                            & (ORIGINAL_DF_SUBJECT_LEVEL.model_version == mv)
                            & (
                                ORIGINAL_DF_SUBJECT_LEVEL.pretraining_type_str
                                == ptv_comparison[1]
                            )
                        ][metric]
                    )

                    if all(
                        [
                            len(population_1) == len(population_2),
                            all([len(p) != 0 for p in [population_1, population_2]]),
                        ]
                    ):
                        # perform test
                        statistical_test = stats.wilcoxon(
                            population_1, population_2, alternative="two-sided"
                        )

                        # print test results
                        string = f"{' '*print_level*level_indent:s}Population 1 ({ptv_comparison[0]}) [mean±std]: {len(population_1)} samples, {np.mean(population_1):0.4f} ± {np.std(population_1):0.4f}"
                        significant_test_analysis_summary.append(string)
                        string = f"{' '*print_level*level_indent:s}Population 2 ({ptv_comparison[1]}) [mean±std]: {len(population_2)} samples, {np.mean(population_2):0.4f} ± {np.std(population_2):0.4f}"
                        significant_test_analysis_summary.append(string)
                        string = f"{' '*print_level*level_indent:s}p-value: {statistical_test[-1]:0.8f} ({'SIGNIFICANT' if statistical_test[-1] <= bonferroni_corrected_significance_thr else 'NOT SIGNIFICANT'})"
                        significant_test_analysis_summary.append(string)
                        # save to the summary of the summary
                        if (
                            statistical_test[-1]
                            <= bonferroni_corrected_significance_thr
                        ):
                            significant_test_analysis_summary_of_the_summary[
                                "significant"
                            ].append(
                                f"{metric}, {mrs}, {mv}, {use_age}, ({ptv_comparison[0]}={np.mean(population_1):0.4f} ± {np.std(population_1):0.4f}, {ptv_comparison[1]}={np.mean(population_2):0.4f} ± {np.std(population_2):0.4f})"
                            )
                        else:
                            significant_test_analysis_summary_of_the_summary[
                                "non_significant"
                            ].append(
                                f"{metric}, {mrs}, {mv}, {use_age}, ({ptv_comparison[0]}={np.mean(population_1):0.4f} ± {np.std(population_1):0.4f}, {ptv_comparison[1]}={np.mean(population_2):0.4f} ± {np.std(population_2):0.4f})"
                            )
                    else:
                        string = f"{' '*print_level*level_indent:s}Skipping statistical test since len populaiton_1=={len(population_1)} and len populaiton_2=={len(population_2)}"
                        significant_test_analysis_summary.append(string)
        significant_test_analysis_summary.append("\n")
    significant_test_analysis_summary.append("\n")


# save also the summary of the summary
for s, s_values in significant_test_analysis_summary_of_the_summary.items():
    string = f"Tests that resulted {s.title()} ({len(s_values)} out of {np.sum([len(v) for v in significant_test_analysis_summary_of_the_summary.values()])})"
    significant_test_analysis_summary.append(string)
    for v in s_values:
        string = v
        significant_test_analysis_summary.append(string)
    string = "\n"
    significant_test_analysis_summary.append(string)

# print summary
for s in significant_test_analysis_summary:
    print(s)

# save to file
summary_file = os.path.join(
    SAVE_PATH, f"Statistical_analysis_comparison_pre_training.txt"
)
with open(summary_file, "w") as f:
    for line in significant_test_analysis_summary:
        f.write(f"{line}\n")

# %% ResNet50 vs ViT_b_16
# SUBJECT WISE COMPARISON BETWEEN TWO MODELS ON THE SAME DATA. HERE WE WILL USE THE NON-PARAMETRIC
# WILCOXON SIGNED-RAKED TEST BETWEEN THE THE POPULATION OF SUBJECT-WISE PERFORMANCE FOR EACH FOLD (50 IN TOTAL
# GIVEN 10 TIMES REPEATED 5 FOLD CROSS VALIDATION) WHEN NOT USING SUBJECT AGE AND WHEN USING SUBJECT AGE.

# Here for each mr sequence, pretraining configuration, input configuration select the two populations
# (ResNet50 and ViT_16_b) and perform the comparison and print the results.

unique_things_to_compare = pd.unique(ORIGINAL_DF_SUBJECT_LEVEL.model_version)
list_of_comparisons = list(itertools.combinations(unique_things_to_compare, 2))
significance_thr = 0.05
bonferroni_corrected_significance_thr = significance_thr / len(list_of_comparisons)

# thinks to keep constant
mr_sequences = ["T1", "T2", "ADC"]
use_ages = [False, True]
pretraining_versions = ["ImageNet", "SimCLR_TCGA", "SimCLR_CBTN"]

# metrics to test
metric_and_text = {
    "overall_precision": {"metric_name": "Precision [0,1]", "metric_range": [0, 1]},
    "overall_recall": {"metric_name": "Recall [0,1]", "metric_range": [0, 1]},
    "overall_accuracy": {"metric_name": "Accuracy [0,1]", "metric_range": [0, 1]},
    "overall_f1-score": {"metric_name": "F1-score [0,1]", "metric_range": [0, 1]},
    "overall_auc": {"metric_name": "AUC [0,1]", "metric_range": [0, 1]},
    "matthews_correlation_coefficient": {
        "metric_name": "Matthews correlation coefficient [-1,1]",
        "metric_range": [0, 1],
    },
}

# things for string formatting
max_len_metric = np.max([len(v) for v in metric_and_text.keys()])
max_len_mr_sequence = np.max([len(v) for v in mr_sequences])
max_len_model_version = np.max([len(v) for v in model_versions])
max_len_pre_t_version = np.max([len(v) for v in pretraining_versions])
max_len_use_age = np.max([len(str(v)) for v in use_ages])
level_indent = 2

# perform analysis and save to file
significant_test_analysis_summary = []
significant_test_analysis_summary_of_the_summary = {
    "significant": [],
    "non_significant": [],
}
significant_test_analysis_summary.append(
    f"Subject-wise statistical analysis of ResNet50 vs ViT_b_16"
)
significant_test_analysis_summary.append(
    f"Using a significance threshold of {significance_thr:0.4f} adjusted using Bonferoni correction for multiple comparisons ({len(list_of_comparisons)} comparisons -> {bonferroni_corrected_significance_thr:0.4f} corrected threshold)."
)

for metric, metric_aus in metric_and_text.items():
    print_level = 0
    string = f"{' '*print_level*level_indent:s}Metric: {metric:{max_len_metric}s}"
    significant_test_analysis_summary.append(string)
    for mrs in mr_sequences:
        print_level = 1
        string = (
            f"{' '*print_level*level_indent:s}MR sequence: {mrs:{max_len_mr_sequence}s}"
        )
        significant_test_analysis_summary.append(string)
        print_level = 2
        for use_age in use_ages:
            print_level = 3
            string = f"{' '*print_level*level_indent:s}Use age: {str(use_age):{max_len_use_age}s}"
            significant_test_analysis_summary.append(string)
            for ptv in pretraining_versions:
                print_level = 4
                string = f"{' '*print_level*level_indent:s}Pre-training version: {ptv:{max_len_pre_t_version}s}"
                significant_test_analysis_summary.append(string)
                print_level = 5

                # get the two populations
                population_1 = list(
                    ORIGINAL_DF_SUBJECT_LEVEL.loc[
                        (ORIGINAL_DF_SUBJECT_LEVEL.use_age == use_age)
                        & (ORIGINAL_DF_SUBJECT_LEVEL.mr_sequence == mrs)
                        & (ORIGINAL_DF_SUBJECT_LEVEL.model_version == "ResNet50")
                        & (ORIGINAL_DF_SUBJECT_LEVEL.pretraining_type_str == ptv)
                    ][metric]
                )
                population_2 = list(
                    ORIGINAL_DF_SUBJECT_LEVEL.loc[
                        (ORIGINAL_DF_SUBJECT_LEVEL.use_age == use_age)
                        & (ORIGINAL_DF_SUBJECT_LEVEL.mr_sequence == mrs)
                        & (ORIGINAL_DF_SUBJECT_LEVEL.model_version == "ViT_b_16")
                        & (ORIGINAL_DF_SUBJECT_LEVEL.pretraining_type_str == ptv)
                    ][metric]
                )

                if all(
                    [
                        len(population_1) == len(population_2),
                        all([len(p) != 0 for p in [population_1, population_2]]),
                    ]
                ):
                    # perform test
                    statistical_test = stats.wilcoxon(
                        population_1, population_2, alternative="two-sided"
                    )

                    # print test results
                    string = f"{' '*print_level*level_indent:s}Population 1 (ResNet50) [mean±std]: {len(population_1)} samples, {np.mean(population_1):0.4f} ± {np.std(population_1):0.4f}"
                    significant_test_analysis_summary.append(string)
                    string = f"{' '*print_level*level_indent:s}Population 2 (ViT_b_16) [mean±std]: {len(population_2)} samples, {np.mean(population_2):0.4f} ± {np.std(population_2):0.4f}"
                    significant_test_analysis_summary.append(string)
                    string = f"{' '*print_level*level_indent:s}p-value: {statistical_test[-1]:0.8f} ({'SIGNIFICANT' if statistical_test[-1] <= bonferroni_corrected_significance_thr else 'NOT SIGNIFICANT'})"
                    significant_test_analysis_summary.append(string)
                    # save to the summary of the summary
                    if statistical_test[-1] <= bonferroni_corrected_significance_thr:
                        significant_test_analysis_summary_of_the_summary[
                            "significant"
                        ].append(
                            f"{metric}, {mrs}, {use_age}, {ptv} (ResNet50={np.mean(population_1):0.4f} ± {np.std(population_1):0.4f}, ViT_b_16={np.mean(population_2):0.4f} ± {np.std(population_2):0.4f})"
                        )
                    else:
                        significant_test_analysis_summary_of_the_summary[
                            "non_significant"
                        ].append(
                            f"{metric}, {mrs}, {use_age}, {ptv} (ResNet50={np.mean(population_1):0.4f} ± {np.std(population_1):0.4f}, ViT_b_16={np.mean(population_2):0.4f} ± {np.std(population_2):0.4f})"
                        )
                else:
                    string = f"{' '*print_level*level_indent:s}Skipping statistical test since len populaiton_1=={len(population_1)} and len populaiton_2=={len(population_2)}"
                    significant_test_analysis_summary.append(string)
        significant_test_analysis_summary.append("\n")
    significant_test_analysis_summary.append("\n")


# save also the summary of the summary
for s, s_values in significant_test_analysis_summary_of_the_summary.items():
    string = f"Tests that resulted {s.title()} ({len(s_values)} out of {np.sum([len(v) for v in significant_test_analysis_summary_of_the_summary.values()])})"
    significant_test_analysis_summary.append(string)
    for v in s_values:
        string = v
        significant_test_analysis_summary.append(string)
    string = "\n"
    significant_test_analysis_summary.append(string)

# print summary
for s in significant_test_analysis_summary:
    print(s)

# save to file
summary_file = os.path.join(SAVE_PATH, f"Statistical_analysis_ResNet50_vs_ViTb16.txt")
with open(summary_file, "w") as f:
    for line in significant_test_analysis_summary:
        f.write(f"{line}\n")


# %% MR SEQUENCES COMPARISON
# SUBJECT WISE COMPARISON BETWEEN MODELS TRAINED ON DIFFERENT MR SEQUENCES. HERE WE ARE USING THE BEST MODEL (MODEL VERSION,
# AGE OR NO AGE, AND PRE-TRAINING) AS REPRESENTATIVES FOR EACH MR SEQUENCE.
# THE NON-PARAMETRIC WILCOXON RANK SUM TEST BETWEEN THE THE POPULATION OF SUBJECT-WISE PERFORMANCES FOR EACH FOLD (50 IN TOTAL
# GIVEN 10 TIMES REPEATED 5 FOLD CROSS VALIDATION). HERE THE POPULATIONS ARE INDEPENDENT AND NOT PAIRED.

unique_things_to_compare = pd.unique(ORIGINAL_DF_SUBJECT_LEVEL.mr_sequence)
list_of_comparisons = list(itertools.combinations(unique_things_to_compare, 2))
significance_thr = 0.05
bonferroni_corrected_significance_thr = significance_thr / len(list_of_comparisons)


# metrics to test
metric_and_text = {
    "overall_precision": {"metric_name": "Precision [0,1]", "metric_range": [0, 1]},
    "overall_recall": {"metric_name": "Recall [0,1]", "metric_range": [0, 1]},
    "overall_accuracy": {"metric_name": "Accuracy [0,1]", "metric_range": [0, 1]},
    "overall_f1-score": {"metric_name": "F1-score [0,1]", "metric_range": [0, 1]},
    "overall_auc": {"metric_name": "AUC [0,1]", "metric_range": [0, 1]},
    "matthews_correlation_coefficient": {
        "metric_name": "Matthews correlation coefficient [-1,1]",
        "metric_range": [0, 1],
    },
}

# things for string formatting
mr_sequences = ["T1", "T2", "ADC"]
use_ages = [False, True]
pretraining_versions = ["ImageNet", "SimCLR_TCGA", "SimCLR_CBTN"]

max_len_metric = np.max([len(v) for v in metric_and_text.keys()])
max_len_mr_sequence = np.max([len(v) for v in mr_sequences])
max_len_model_version = np.max([len(v) for v in model_versions])
max_len_pre_t_version = np.max([len(v) for v in pretraining_versions])
max_len_use_age = np.max([len(str(v)) for v in use_ages])
level_indent = 2

# perform analysis and save to file
significant_test_analysis_summary = []
significant_test_analysis_summary_of_the_summary = {
    "significant": [],
    "non_significant": [],
}
significant_test_analysis_summary.append(
    f"Subject-wise comparison between the best performing models for each MR sequence.\nUsing the Non-parametric Wilcoxon rank sum test since the populations are not paired and not normally distributed."
)
significant_test_analysis_summary.append(
    f"Using a significance threshold of {significance_thr:0.4f} adjusted using Bonferoni correction for multiple comparisons ({len(list_of_comparisons)} comparisons -> {bonferroni_corrected_significance_thr:0.4f} corrected threshold)."
)

for metric, metric_aus in metric_and_text.items():
    print_level = 0
    string = f"{' '*print_level*level_indent:s}Metric: {metric:{max_len_metric}s}"
    significant_test_analysis_summary.append(string)

    # compress information for this metric to make it easier to get the indexes of the best performing model
    COMPRESSED_DF_SUBJECT_LEVEL = (
        ORIGINAL_DF_SUBJECT_LEVEL.groupby(
            ["mr_sequence", "model_version", "use_age", "pretraining_type_str"]
        )
        .apply(lambda x: x.matthews_correlation_coefficient.mean())
        .reset_index(name=metric)
    )

    # where to store the information for each mr sequence and metric
    best_model_performance_settings_and_populations = {}

    for mrs in unique_things_to_compare:
        # get the best model for this mr sequence (based on average performance) and save parameters
        aus_df = COMPRESSED_DF_SUBJECT_LEVEL.iloc[
            COMPRESSED_DF_SUBJECT_LEVEL.loc[
                COMPRESSED_DF_SUBJECT_LEVEL.mr_sequence == mrs
            ].idxmax(numeric_only=True)[metric]
        ]

        # save values
        best_model_performance_settings_and_populations[mrs] = {
            "model_version": aus_df.model_version,
            "use_age": aus_df.use_age,
            "pretraining_type_str": aus_df.pretraining_type_str,
            "model_settings_string": "__".join(
                [
                    mrs,
                    aus_df.model_version,
                    str(aus_df.use_age),
                    aus_df.pretraining_type_str,
                ]
            ),
            "population_values": list(
                ORIGINAL_DF_SUBJECT_LEVEL.loc[
                    (ORIGINAL_DF_SUBJECT_LEVEL.use_age == aus_df.use_age)
                    & (ORIGINAL_DF_SUBJECT_LEVEL.mr_sequence == mrs)
                    & (ORIGINAL_DF_SUBJECT_LEVEL.model_version == aus_df.model_version)
                    & (
                        ORIGINAL_DF_SUBJECT_LEVEL.pretraining_type_str
                        == aus_df.pretraining_type_str
                    )
                ][metric]
            ),
        }

    # now that all the information is available, perform comparisons
    for ptv_comparison in list_of_comparisons:
        print_level = 1
        string = f"{' '*print_level*level_indent:s}Comparison: {'-vs-'.join(ptv_comparison):10s}"
        significant_test_analysis_summary.append(string)
        print_level = 2

        # get the two populations
        population_1 = best_model_performance_settings_and_populations[
            ptv_comparison[0]
        ]["population_values"]
        population_2 = best_model_performance_settings_and_populations[
            ptv_comparison[1]
        ]["population_values"]

        if all(
            [
                len(population_1) == len(population_2),
                all([len(p) != 0 for p in [population_1, population_2]]),
            ]
        ):
            # perform test
            statistical_test = stats.ranksums(
                population_1, population_2, alternative="two-sided"
            )

            # print test results
            string = f"{' '*print_level*level_indent:s}Population 1 ({best_model_performance_settings_and_populations[ptv_comparison[0]]['model_settings_string']}) [mean±std]: {len(population_1)} samples, {np.mean(population_1):0.4f} ± {np.std(population_1):0.4f}"
            significant_test_analysis_summary.append(string)
            string = f"{' '*print_level*level_indent:s}Population 2 ({best_model_performance_settings_and_populations[ptv_comparison[1]]['model_settings_string']}) [mean±std]: {len(population_2)} samples, {np.mean(population_2):0.4f} ± {np.std(population_2):0.4f}"
            significant_test_analysis_summary.append(string)
            string = f"{' '*print_level*level_indent:s}p-value: {statistical_test[-1]:0.10f} ({'SIGNIFICANT' if statistical_test[-1] <= bonferroni_corrected_significance_thr else 'NOT SIGNIFICANT'})"
            significant_test_analysis_summary.append(string)
            # save to the summary of the summary
            if statistical_test[-1] <= bonferroni_corrected_significance_thr:
                significant_test_analysis_summary_of_the_summary["significant"].append(
                    f"{metric}: {best_model_performance_settings_and_populations[ptv_comparison[0]]['model_settings_string']}={np.mean(population_1):0.4f} ± {np.std(population_1):0.4f}, {best_model_performance_settings_and_populations[ptv_comparison[1]]['model_settings_string']}={np.mean(population_2):0.4f} ± {np.std(population_2):0.4f}"
                )
            else:
                significant_test_analysis_summary_of_the_summary[
                    "non_significant"
                ].append(
                    f"{metric}: {best_model_performance_settings_and_populations[ptv_comparison[0]]['model_settings_string']}={np.mean(population_1):0.4f} ± {np.std(population_1):0.4f}, {best_model_performance_settings_and_populations[ptv_comparison[1]]['model_settings_string']}={np.mean(population_2):0.4f} ± {np.std(population_2):0.4f}"
                )
        else:
            string = f"{' '*print_level*level_indent:s}Skipping statistical test since len populaiton_1=={len(population_1)} and len populaiton_2=={len(population_2)}"
            significant_test_analysis_summary.append(string)
    significant_test_analysis_summary.append("\n")

# save also the summary of the summary
for s, s_values in significant_test_analysis_summary_of_the_summary.items():
    string = f"Tests that resulted {s.title()} ({len(s_values)} out of {np.sum([len(v) for v in significant_test_analysis_summary_of_the_summary.values()])})"
    significant_test_analysis_summary.append(string)
    for v in s_values:
        string = v
        significant_test_analysis_summary.append(string)
    string = "\n"
    significant_test_analysis_summary.append(string)

# print summary
for s in significant_test_analysis_summary:
    print(s)

# save to file
summary_file = os.path.join(
    SAVE_PATH, f"Statistical_analysis_comparison_MR_sequences.txt"
)
with open(summary_file, "w") as f:
    for line in significant_test_analysis_summary:
        f.write(f"{line}\n")

# %%
