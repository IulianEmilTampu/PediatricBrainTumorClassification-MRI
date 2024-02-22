# %%
"""
Script that uses the aggregated .csv file of the model performances to plot a table with the models performances for the different configurations.
Here we want to plot the following (like in the table for the manuscript) for SLICE_WISE & SUBJECT_WISE:
SLICE_WISE & SUBJECT_WISE 

MR_sequence | MODEL | Use_age | Pre-training_strategy | Pre-training_dataset | MCC [mean+-std] | Accuracy [mean+-std] | AUC [mean+-std] | Class_wise_F1 [mean+-std] | Class_wise_AUC [mean+-std]  

To easily plot, we use a pandas dataframe and then use groupby to plot
"""
import os
import pandas as pd
import numpy as np
import pathlib


# %% UTILITIES
def make_summary_string(x):
    # get all the information needed
    mrs = x.mr_sequence
    model = x.model_version
    prt = x.pretraining_type_str

    return f"{mrs}\n{model}\n{prt}"


def replace_all(string, dict_for_replacement: dict = {"[": "", "]": "", " ": ","}):
    for r, s in dict_for_replacement.items():
        string = string.replace(r, s)
    return string


def convert_str_list_to_float(x):
    x = replace_all(x)
    x = x.split(",")
    # remove extra white spaces
    x = [f for f in x if len(f) != 0]
    # there are some ADC splits where the EP class is missing (the stratification did not put a class there). Add a NaN value in those cases
    if len(x) == 2:
        x = [x[0], np.nan, x[1]]
    return np.array(x, dtype=np.float32)


# %% OLD IMPLEMENTATION
SUMMARY_FILE_PATH = "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/evaluation_results/Evaluation_20240212/summary_evaluation_aggregated.csv"
SAVE_PATH = pathlib.Path(os.path.dirname(SUMMARY_FILE_PATH))
ORIGINAL_DF = pd.read_csv(SUMMARY_FILE_PATH)
SET = "test"

# use model with 0.5 fine tuning
ORIGINAL_DF = ORIGINAL_DF.loc[ORIGINAL_DF.fine_tuning == 0.5]
# use plot results for a 3-class classification problem
ORIGINAL_DF = ORIGINAL_DF.loc[ORIGINAL_DF.nbr_classes == 3]
# only use the specified set performances
ORIGINAL_DF = ORIGINAL_DF.loc[ORIGINAL_DF.evaluation_set == SET]

# ## adjust some values for convinience

# fix the pre_training dataset for the cases where pretraining is false. Set pretraining dataset to imagenet
ORIGINAL_DF.loc[ORIGINAL_DF.pretraining == False, "pretraining_dataset"] = "imagenet"

# convert the string values for the per-class performances into float values
for metric in ["accuracy", "precision", "recall", "f1-score"]:
    ORIGINAL_DF[metric] = ORIGINAL_DF[metric].apply(
        lambda x: convert_str_list_to_float(x)
    )

# define levels
MR_SEQUENCES = ["T1", "T2", "ADC"]
MODELS = ["ResNet50", "ViT_b_16"]
USE_AGE = [False, True]
PRE_TRAINING = [False, True]
PRE_TRAINING_DATASET = ["imagenet", "tcga", "cbtn"]
CLASSES = ["ASTR", "EP", "MED"]

# define spaces in the string for each of these
buffer = 1
delimiter = " "
mr_sequence_string_space = delimiter * (np.max([len(s) for s in MR_SEQUENCES]) + buffer)
models_string_space = delimiter * (np.max([len(s) for s in MODELS]) + buffer)
age_string_space = delimiter * (np.max([len(str(s)) for s in USE_AGE]) + buffer)
pre_training_string_space = delimiter * (
    np.max([len(str(s)) for s in PRE_TRAINING]) + buffer
)
pre_training_dataset_string_space = delimiter * (
    np.max([len(str(s)) for s in PRE_TRAINING_DATASET]) + buffer
)
metric_space = delimiter * 20
class_space = delimiter * (np.max([len(s) for s in CLASSES]) + buffer)
pm_symbol = f" \u00B1 "


# %% SLICE-WISE
# Loop through the different levels

filter_for_slice_performance = [f"pred_fold_{i+1}" for i in range(10)]
list_strings_to_plot = []

for idx_mrs, mrs in enumerate(MR_SEQUENCES):
    print_mrs = True
    for idx_model, model in enumerate(MODELS):
        print_model = True
        for idx_age, use_age in enumerate(USE_AGE):
            print_use_age = True
            for idx_pre_t, pre_training in enumerate(PRE_TRAINING):
                print_pre_t = True
                if pre_training:
                    PRE_TRAINING_DATASET = ["tcga", "cbtn"]
                else:
                    PRE_TRAINING_DATASET = ["imagenet"]
                for idx_pre_t_d, pre_training_dataset in enumerate(
                    PRE_TRAINING_DATASET
                ):
                    print_pre_t_d = True
                    # filter dataframe to get the information for this model configuration
                    performance_info = ORIGINAL_DF.loc[
                        (ORIGINAL_DF.mr_sequence == mrs)
                        & (ORIGINAL_DF.model_version == model)
                        & (ORIGINAL_DF.use_age == use_age)
                        & (ORIGINAL_DF.pretraining == pre_training)
                        & (ORIGINAL_DF.pretraining_dataset == pre_training_dataset)
                    ]

                    # use only the slice-level performance
                    performance_info = performance_info[
                        performance_info.performance_over.isin(
                            filter_for_slice_performance
                        )
                    ]
                    if len(performance_info) > 0:
                        print_metric = True

                        # collect overall performance information (not class dependent)
                        mean_mcc = np.nanmean(
                            performance_info.matthews_correlation_coefficient
                        )
                        std_mcc = np.nanstd(
                            performance_info.matthews_correlation_coefficient
                        )
                        mcc_string = f"{mean_mcc:0.4f}{pm_symbol}{std_mcc:0.4f}"

                        mean_acc = np.nanmean(performance_info.overall_accuracy)
                        std_acc = np.nanstd(performance_info.overall_accuracy)
                        acc_string = f"{mean_acc:0.4f}{pm_symbol}{std_acc:0.4f}"

                        mean_auc = np.nanmean(performance_info.overall_auc)
                        std_auc = np.nanstd(performance_info.overall_auc)
                        auc_string = f"{mean_auc:0.4f}{pm_symbol}{std_auc:0.4f}"

                        # class-wise mean values
                        class_wise_mean_f1 = np.nanmean(
                            np.vstack(performance_info["f1-score"]), axis=0
                        )
                        class_wise_std_f1 = np.nanstd(
                            np.vstack(performance_info["f1-score"]), axis=0
                        )

                        class_wise_mean_precision = np.nanmean(
                            np.vstack(performance_info["precision"]), axis=0
                        )
                        class_wise_std_precision = np.nanstd(
                            np.vstack(performance_info["precision"]), axis=0
                        )

                        class_wise_mean_recall = np.nanmean(
                            np.vstack(performance_info["recall"]), axis=0
                        )
                        class_wise_std_recall = np.nanstd(
                            np.vstack(performance_info["recall"]), axis=0
                        )

                        # class-wise information
                        for idc, class_name in enumerate(CLASSES):
                            class_wise_metric_strings = []
                            for metric, values in zip(
                                ["precision", "recall", "f1-score"],
                                [
                                    (
                                        class_wise_mean_precision,
                                        class_wise_std_precision,
                                    ),
                                    (class_wise_mean_recall, class_wise_std_recall),
                                    (class_wise_mean_f1, class_wise_std_f1),
                                ],
                            ):
                                class_wise_metric_string = f"{values[0][idc]:0.4f}{pm_symbol}{values[1][idc]:0.4f}"
                                class_wise_metric_strings.append(
                                    f"{class_wise_metric_string:{len(metric_space)}s}"
                                )

                            # build string to print
                            mrs_sequence_string = f"{mrs if print_mrs else mr_sequence_string_space:{len(mr_sequence_string_space)}s}"
                            model_string = f"{model if print_model else models_string_space:{len(models_string_space)}s}"
                            use_age_string = f"{str(use_age) if print_use_age else age_string_space:{len(age_string_space)}s}"
                            pre_t_string = f"{str(pre_training) if print_pre_t else pre_training_string_space:{len(pre_training_string_space)}s}"
                            pre_t_d_string = f"{pre_training_dataset if print_pre_t_d else pre_training_dataset_string_space:{len(pre_training_dataset_string_space)}s}"
                            mcc_string = f"{mcc_string if print_metric else metric_space:{len(metric_space)}s}"
                            acc_string = f"{acc_string if print_metric else metric_space:{len(metric_space)}s}"
                            auc_string = f"{auc_string if print_metric else metric_space:{len(metric_space)}s}"
                            class_name_string = f"{class_name:{len(class_space)}s}"

                            # this is quite tricky, but bare with me :)
                            list_strings_to_plot.append(
                                "".join(
                                    [
                                        mrs_sequence_string,
                                        model_string,
                                        use_age_string,
                                        pre_t_string,
                                        pre_t_d_string,
                                        mcc_string,
                                        acc_string,
                                        auc_string,
                                        class_name_string,
                                        " ".join(class_wise_metric_strings),
                                    ]
                                )
                            )

                            # (
                            #     print_mrs,
                            #     print_model,
                            #     print_use_age,
                            #     print_pre_t,
                            #     print_pre_t_d,
                            #     print_metric,
                            # ) = (False, False, False, False, False, False)

                    else:
                        # build empty string since model is still training
                        not_available_string = "NaN"
                        for idc, class_name in enumerate(CLASSES):
                            class_wise_metric_strings = []
                            for metric, values in zip(
                                ["precision", "recall", "f1-score"],
                                [
                                    (not_available_string, not_available_string),
                                    (not_available_string, not_available_string),
                                    (not_available_string, not_available_string),
                                ],
                            ):
                                class_wise_metric_string = (
                                    f"{values[0]}{pm_symbol}{values[1]}"
                                )
                                class_wise_metric_strings.append(
                                    f"{class_wise_metric_string:{len(metric_space)}s}"
                                )

                            # build string to print
                            mrs_sequence_string = f"{mrs if print_mrs else mr_sequence_string_space:{len(mr_sequence_string_space)}s}"
                            model_string = f"{model if print_model else models_string_space:{len(models_string_space)}s}"
                            use_age_string = f"{str(use_age) if print_use_age else age_string_space:{len(age_string_space)}s}"
                            pre_t_string = f"{('SimCLR' if pre_training else '/') if print_pre_t else pre_training_string_space:{len(pre_training_string_space)}s}"
                            pre_t_d_string = f"{pre_training_dataset if print_pre_t_d else pre_training_dataset_string_space:{len(pre_training_dataset_string_space)}s}"
                            mcc_string = f"{not_available_string if print_metric else metric_space:{len(metric_space)}s}"
                            acc_string = f"{not_available_string if print_metric else metric_space:{len(metric_space)}s}"
                            auc_string = f"{not_available_string if print_metric else metric_space:{len(metric_space)}s}"
                            class_name_string = f"{class_name:{len(class_space)}s}"

                            # this is quite tricky, but bare with me :)
                            list_strings_to_plot.append(
                                "".join(
                                    [
                                        mrs_sequence_string,
                                        model_string,
                                        use_age_string,
                                        pre_t_string,
                                        pre_t_d_string,
                                        mcc_string,
                                        acc_string,
                                        auc_string,
                                        class_name_string,
                                        " ".join(class_wise_metric_strings),
                                    ]
                                )
                            )

                            # (
                            #     print_mrs,
                            #     print_model,
                            #     print_use_age,
                            #     print_pre_t,
                            #     print_pre_t_d,
                            #     print_metric,
                            # ) = (False, False, False, False, False, False)

for line in list_strings_to_plot:
    print(line)


# %% SUBJECT-WISE
# Loop through the different levels
filter_for_slice_performance = [f"subject_ensemble_pred_fold_{i+1}" for i in range(10)]
list_strings_to_plot = []

for idx_mrs, mrs in enumerate(MR_SEQUENCES):
    print_mrs = True
    for idx_model, model in enumerate(MODELS):
        print_model = True
        for idx_age, use_age in enumerate(USE_AGE):
            print_use_age = True
            for idx_pre_t, pre_training in enumerate(PRE_TRAINING):
                print_pre_t = True
                if pre_training:
                    PRE_TRAINING_DATASET = ["tcga", "cbtn"]
                else:
                    PRE_TRAINING_DATASET = ["imagenet"]
                for idx_pre_t_d, pre_training_dataset in enumerate(
                    PRE_TRAINING_DATASET
                ):
                    print_pre_t_d = True
                    # filter dataframe to get the information for this model configuration
                    performance_info = ORIGINAL_DF.loc[
                        (ORIGINAL_DF.mr_sequence == mrs)
                        & (ORIGINAL_DF.model_version == model)
                        & (ORIGINAL_DF.use_age == use_age)
                        & (ORIGINAL_DF.pretraining == pre_training)
                        & (ORIGINAL_DF.pretraining_dataset == pre_training_dataset)
                    ]

                    # use only the slice-level performance
                    performance_info = performance_info[
                        performance_info.performance_over.isin(
                            filter_for_slice_performance
                        )
                    ]
                    if len(performance_info) > 0:
                        print_metric = True

                        # collect overall performance information (not class dependent)
                        mean_mcc = np.nanmean(
                            performance_info.matthews_correlation_coefficient
                        )
                        std_mcc = np.nanstd(
                            performance_info.matthews_correlation_coefficient
                        )
                        mcc_string = f"{mean_mcc:0.4f}{pm_symbol}{std_mcc:0.4f}"

                        mean_acc = np.nanmean(performance_info.overall_accuracy)
                        std_acc = np.nanstd(performance_info.overall_accuracy)
                        acc_string = f"{mean_acc:0.4f}{pm_symbol}{std_acc:0.4f}"

                        mean_auc = np.nanmean(performance_info.overall_auc)
                        std_auc = np.nanstd(performance_info.overall_auc)
                        auc_string = f"{mean_auc:0.4f}{pm_symbol}{std_auc:0.4f}"

                        # class-wise mean values
                        class_wise_mean_f1 = np.nanmean(
                            np.vstack(performance_info["f1-score"]), axis=0
                        )
                        class_wise_std_f1 = np.nanstd(
                            np.vstack(performance_info["f1-score"]), axis=0
                        )

                        class_wise_mean_precision = np.nanmean(
                            np.vstack(performance_info["precision"]), axis=0
                        )
                        class_wise_std_precision = np.nanstd(
                            np.vstack(performance_info["precision"]), axis=0
                        )

                        class_wise_mean_recall = np.nanmean(
                            np.vstack(performance_info["recall"]), axis=0
                        )
                        class_wise_std_recall = np.nanstd(
                            np.vstack(performance_info["recall"]), axis=0
                        )

                        # class-wise information
                        for idc, class_name in enumerate(CLASSES):
                            class_wise_metric_strings = []
                            for metric, values in zip(
                                ["precision", "recall", "f1-score"],
                                [
                                    (
                                        class_wise_mean_precision,
                                        class_wise_std_precision,
                                    ),
                                    (class_wise_mean_recall, class_wise_std_recall),
                                    (class_wise_mean_f1, class_wise_std_f1),
                                ],
                            ):
                                class_wise_metric_string = f"{values[0][idc]:0.4f}{pm_symbol}{values[1][idc]:0.4f}"
                                class_wise_metric_strings.append(
                                    f"{class_wise_metric_string:{len(metric_space)}s}"
                                )

                            # build string to print
                            mrs_sequence_string = f"{mrs if print_mrs else mr_sequence_string_space:{len(mr_sequence_string_space)}s}"
                            model_string = f"{model if print_model else models_string_space:{len(models_string_space)}s}"
                            use_age_string = f"{str(use_age) if print_use_age else age_string_space:{len(age_string_space)}s}"
                            pre_t_string = f"{('SimCLR' if pre_training else '/') if print_pre_t else pre_training_string_space:{len(pre_training_string_space)}s}"
                            pre_t_d_string = f"{pre_training_dataset if print_pre_t_d else pre_training_dataset_string_space:{len(pre_training_dataset_string_space)}s}"
                            mcc_string = f"{mcc_string if print_metric else metric_space:{len(metric_space)}s}"
                            acc_string = f"{acc_string if print_metric else metric_space:{len(metric_space)}s}"
                            auc_string = f"{auc_string if print_metric else metric_space:{len(metric_space)}s}"
                            class_name_string = f"{class_name:{len(class_space)}s}"

                            # this is quite tricky, but bare with me :)
                            list_strings_to_plot.append(
                                "".join(
                                    [
                                        mrs_sequence_string,
                                        model_string,
                                        use_age_string,
                                        pre_t_string,
                                        pre_t_d_string,
                                        mcc_string,
                                        acc_string,
                                        auc_string,
                                        class_name_string,
                                        " ".join(class_wise_metric_strings),
                                    ]
                                )
                            )

                            (
                                print_mrs,
                                print_model,
                                print_use_age,
                                print_pre_t,
                                print_pre_t_d,
                                print_metric,
                            ) = (False, False, False, False, False, False)

                    else:
                        # build empty string since model is still training
                        not_available_string = "NaN"
                        for idc, class_name in enumerate(CLASSES):
                            class_wise_metric_strings = []
                            for metric, values in zip(
                                ["precision", "recall", "f1-score"],
                                [
                                    (not_available_string, not_available_string),
                                    (not_available_string, not_available_string),
                                    (not_available_string, not_available_string),
                                ],
                            ):
                                class_wise_metric_string = (
                                    f"{values[0]}{pm_symbol}{values[1]}"
                                )
                                class_wise_metric_strings.append(
                                    f"{class_wise_metric_string:{len(metric_space)}s}"
                                )

                            # build string to print
                            mrs_sequence_string = f"{mrs if print_mrs else mr_sequence_string_space:{len(mr_sequence_string_space)}s}"
                            model_string = f"{model if print_model else models_string_space:{len(models_string_space)}s}"
                            use_age_string = f"{str(use_age) if print_use_age else age_string_space:{len(age_string_space)}s}"
                            pre_t_string = f"{str(pre_training) if print_pre_t else pre_training_string_space:{len(pre_training_string_space)}s}"
                            pre_t_d_string = f"{pre_training_dataset if print_pre_t_d else pre_training_dataset_string_space:{len(pre_training_dataset_string_space)}s}"
                            mcc_string = f"{not_available_string if print_metric else metric_space:{len(metric_space)}s}"
                            acc_string = f"{not_available_string if print_metric else metric_space:{len(metric_space)}s}"
                            auc_string = f"{not_available_string if print_metric else metric_space:{len(metric_space)}s}"
                            class_name_string = f"{class_name:{len(class_space)}s}"

                            # this is quite tricky, but bare with me :)
                            list_strings_to_plot.append(
                                "".join(
                                    [
                                        mrs_sequence_string,
                                        model_string,
                                        use_age_string,
                                        pre_t_string,
                                        pre_t_d_string,
                                        mcc_string,
                                        acc_string,
                                        auc_string,
                                        class_name_string,
                                        " ".join(class_wise_metric_strings),
                                    ]
                                )
                            )

                            (
                                print_mrs,
                                print_model,
                                print_use_age,
                                print_pre_t,
                                print_pre_t_d,
                                print_metric,
                            ) = (False, False, False, False, False, False)

for line in list_strings_to_plot:
    print(line)

# %% MAKE LATEX TABLE_CODE SUBJECT-WISE

# define spaces in the string for each of these
buffer = 1
delimiter = " "
# mr_sequence_string_space = delimiter * (np.max([len(s) for s in MR_SEQUENCES]) + buffer)
# models_string_space = delimiter * (np.max([len(s) for s in MODELS]) + buffer)
# age_string_space = delimiter * (np.max([len(str(s)) for s in USE_AGE]) + buffer)
# pre_training_string_space = delimiter * (
#     np.max([len(str(s)) for s in PRE_TRAINING]) + buffer
# )
# pre_training_dataset_string_space = delimiter * (
#     np.max([len(str(s)) for s in PRE_TRAINING_DATASET]) + buffer
# )
# metric_space = delimiter * 20
# class_space = delimiter * (np.max([len(s) for s in CLASSES]) + buffer)
mr_sequence_string_space = delimiter
models_string_space = delimiter
age_string_space = delimiter
pre_training_string_space = delimiter
pre_training_dataset_string_space = delimiter
metric_space = delimiter
class_space = delimiter
pm_symbol = f" \u00B1 "


# Loop through the different levels
filter_for_slice_performance = [f"subject_ensemble_pred_fold_{i+1}" for i in range(10)]
list_strings_to_plot = []

for idx_mrs, mrs in enumerate(MR_SEQUENCES):
    print_mrs = True
    for idx_model, model in enumerate(MODELS):
        print_model = True
        for idx_age, use_age in enumerate(USE_AGE):
            print_use_age = True
            for idx_pre_t, pre_training in enumerate(PRE_TRAINING):
                print_pre_t = True
                if pre_training:
                    PRE_TRAINING_DATASET = ["tcga", "cbtn"]
                else:
                    PRE_TRAINING_DATASET = ["imagenet"]
                for idx_pre_t_d, pre_training_dataset in enumerate(
                    PRE_TRAINING_DATASET
                ):
                    print_pre_t_d = True
                    # filter dataframe to get the information for this model configuration
                    performance_info = ORIGINAL_DF.loc[
                        (ORIGINAL_DF.mr_sequence == mrs)
                        & (ORIGINAL_DF.model_version == model)
                        & (ORIGINAL_DF.use_age == use_age)
                        & (ORIGINAL_DF.pretraining == pre_training)
                        & (ORIGINAL_DF.pretraining_dataset == pre_training_dataset)
                    ]

                    # use only the slice-level performance
                    performance_info = performance_info[
                        performance_info.performance_over.isin(
                            filter_for_slice_performance
                        )
                    ]
                    if len(performance_info) > 0:
                        print_metric = True

                        # collect overall performance information (not class dependent)
                        mean_mcc = np.nanmean(
                            performance_info.matthews_correlation_coefficient
                        )
                        std_mcc = np.nanstd(
                            performance_info.matthews_correlation_coefficient
                        )
                        mcc_string = f"{mean_mcc:0.4f}{pm_symbol}{std_mcc:0.4f}"

                        mean_acc = np.nanmean(performance_info.overall_accuracy)
                        std_acc = np.nanstd(performance_info.overall_accuracy)
                        acc_string = f"{mean_acc:0.4f}{pm_symbol}{std_acc:0.4f}"

                        mean_auc = np.nanmean(performance_info.overall_auc)
                        std_auc = np.nanstd(performance_info.overall_auc)
                        auc_string = f"{mean_auc:0.4f}{pm_symbol}{std_auc:0.4f}"

                        # class-wise mean values
                        class_wise_mean_f1 = np.nanmean(
                            np.vstack(performance_info["f1-score"]), axis=0
                        )
                        class_wise_std_f1 = np.nanstd(
                            np.vstack(performance_info["f1-score"]), axis=0
                        )

                        class_wise_mean_precision = np.nanmean(
                            np.vstack(performance_info["precision"]), axis=0
                        )
                        class_wise_std_precision = np.nanstd(
                            np.vstack(performance_info["precision"]), axis=0
                        )

                        class_wise_mean_recall = np.nanmean(
                            np.vstack(performance_info["recall"]), axis=0
                        )
                        class_wise_std_recall = np.nanstd(
                            np.vstack(performance_info["recall"]), axis=0
                        )

                        # class-wise information
                        for idc, class_name in enumerate(CLASSES):
                            class_wise_metric_strings = []
                            for metric, values in zip(
                                ["precision", "recall", "f1-score"],
                                [
                                    (
                                        class_wise_mean_precision,
                                        class_wise_std_precision,
                                    ),
                                    (class_wise_mean_recall, class_wise_std_recall),
                                    (class_wise_mean_f1, class_wise_std_f1),
                                ],
                            ):
                                class_wise_metric_string = f"\\Block{{1-1}}<\\{'f'}ootnotesize>{{${values[0][idc]:0.4f} \\pm {values[1][idc]:0.4f}$}}"
                                class_wise_metric_strings.append(
                                    f"{class_wise_metric_string:{len(metric_space)}s}"
                                )

                            # build string to print (using LaTex codes for the different columns)
                            mrs_aus = f"\\Block{{36-1}}{{{mrs}}}"
                            model_aus = f"\\Block{{18-1}}{{\\rotnd{{{model}}}}}"
                            use_age_aus = f"\\Block{{9-1}}{{{str(use_age)}}}"
                            pre_t_aus = f'\\Block{{{6 if pre_training else 3}-1}}{{{str(("SimCLR" if pre_training else "/") )}}}'
                            pre_t_d_aus = (
                                f"\\Block{{3-1}}{{{pre_training_dataset.title()}}}"
                            )

                            mcc_aus = f"\\Block{{3-1}}{{${mean_mcc:0.4f} \\pm {std_mcc:0.4f}$}}"
                            acc_aus = f"\\Block{{3-1}}{{${mean_acc:0.4f} \\pm {std_acc:0.4f}$}}"
                            auc_aus = f"\\Block{{3-1}}{{${mean_auc:0.4f} \\pm {std_auc:0.4f}$}}"

                            class_name_aus = f"\\Block{{1-1}}{{{class_name}}}"

                            mrs_sequence_string = f"{mrs_aus if print_mrs else mr_sequence_string_space:{len(mr_sequence_string_space)}s}"
                            model_string = f"{model_aus if print_model else models_string_space:{len(models_string_space)}s}"
                            use_age_string = f"{use_age_aus if print_use_age else age_string_space:{len(age_string_space)}s}"
                            pre_t_string = f"{pre_t_aus if print_pre_t else pre_training_string_space:{len(pre_training_string_space)}s}"
                            pre_t_d_string = f"{pre_t_d_aus if print_pre_t_d else pre_training_dataset_string_space:{len(pre_training_dataset_string_space)}s}"
                            mcc_string = f"{mcc_aus if print_metric else metric_space:{len(metric_space)}s}"
                            acc_string = f"{acc_aus if print_metric else metric_space:{len(metric_space)}s}"
                            auc_string = f"{auc_aus if print_metric else metric_space:{len(metric_space)}s}"
                            class_name_string = f"{class_name_aus}"

                            # this is quite tricky, but bare with me :)
                            list_strings_to_plot.append(
                                "&".join(
                                    [
                                        mrs_sequence_string,
                                        model_string,
                                        use_age_string,
                                        pre_t_string,
                                        pre_t_d_string,
                                        mcc_string,
                                        acc_string,
                                        auc_string,
                                        class_name_string,
                                        "&".join(class_wise_metric_strings),
                                    ]
                                )
                                + f"\\\\"
                            )

                            (
                                print_mrs,
                                print_model,
                                print_use_age,
                                print_pre_t,
                                print_pre_t_d,
                                print_metric,
                            ) = (False, False, False, False, False, False)

                    else:
                        # build empty string since model is still training
                        not_available_string = "NaN"
                        for idc, class_name in enumerate(CLASSES):
                            class_wise_metric_strings = []
                            for metric, values in zip(
                                ["precision", "recall", "f1-score"],
                                [
                                    (not_available_string, not_available_string),
                                    (not_available_string, not_available_string),
                                    (not_available_string, not_available_string),
                                ],
                            ):
                                class_wise_metric_string = (
                                    f"{values[0]}{pm_symbol}{values[1]}"
                                )
                                class_wise_metric_strings.append(
                                    f"{class_wise_metric_string:{len(metric_space)}s}"
                                )

                            # build string to print
                            mrs_sequence_string = f"{mrs if print_mrs else mr_sequence_string_space:{len(mr_sequence_string_space)}s}"
                            model_string = f"{model if print_model else models_string_space:{len(models_string_space)}s}"
                            use_age_string = f"{str(use_age) if print_use_age else age_string_space:{len(age_string_space)}s}"
                            pre_t_string = f"{str(pre_training) if print_pre_t else pre_training_string_space:{len(pre_training_string_space)}s}"
                            pre_t_d_string = f"{pre_training_dataset if print_pre_t_d else pre_training_dataset_string_space:{len(pre_training_dataset_string_space)}s}"
                            mcc_string = f"{not_available_string if print_metric else metric_space:{len(metric_space)}s}"
                            acc_string = f"{not_available_string if print_metric else metric_space:{len(metric_space)}s}"
                            auc_string = f"{not_available_string if print_metric else metric_space:{len(metric_space)}s}"
                            class_name_string = f"{class_name:{len(class_space)}s}"

                            # this is quite tricky, but bare with me :)
                            list_strings_to_plot.append(
                                "".join(
                                    [
                                        mrs_sequence_string,
                                        model_string,
                                        use_age_string,
                                        pre_t_string,
                                        pre_t_d_string,
                                        mcc_string,
                                        acc_string,
                                        auc_string,
                                        class_name_string,
                                        " ".join(class_wise_metric_strings),
                                    ]
                                )
                                + f"\\\\"
                            )

                            (
                                print_mrs,
                                print_model,
                                print_use_age,
                                print_pre_t,
                                print_pre_t_d,
                                print_metric,
                            ) = (False, False, False, False, False, False)

for line in list_strings_to_plot:
    print(line)

# %% LATEX-LIKE WITHOUT CLASS WISE METRICS
# Loop through the different levels
filter_for_slice_performance = [f"subject_ensemble_pred_fold_{i+1}" for i in range(10)]
list_strings_to_plot = []

for idx_mrs, mrs in enumerate(MR_SEQUENCES):
    print_mrs = True
    for idx_model, model in enumerate(MODELS):
        print_model = True
        for idx_age, use_age in enumerate(USE_AGE):
            print_use_age = True
            for idx_pre_t, pre_training in enumerate(PRE_TRAINING):
                print_pre_t = True
                if pre_training:
                    PRE_TRAINING_DATASET = ["tcga", "cbtn"]
                else:
                    PRE_TRAINING_DATASET = ["imagenet"]
                for idx_pre_t_d, pre_training_dataset in enumerate(
                    PRE_TRAINING_DATASET
                ):
                    print_pre_t_d = True
                    # filter dataframe to get the information for this model configuration
                    performance_info = ORIGINAL_DF.loc[
                        (ORIGINAL_DF.mr_sequence == mrs)
                        & (ORIGINAL_DF.model_version == model)
                        & (ORIGINAL_DF.use_age == use_age)
                        & (ORIGINAL_DF.pretraining == pre_training)
                        & (ORIGINAL_DF.pretraining_dataset == pre_training_dataset)
                    ]

                    # use only the slice-level performance
                    performance_info = performance_info[
                        performance_info.performance_over.isin(
                            filter_for_slice_performance
                        )
                    ]
                    if len(performance_info) > 0:
                        print_metric = True

                        # collect overall performance information (not class dependent)
                        mean_mcc = np.nanmean(
                            performance_info.matthews_correlation_coefficient
                        )
                        std_mcc = np.nanstd(
                            performance_info.matthews_correlation_coefficient
                        )
                        mcc_string = f"{mean_mcc:0.4f}{pm_symbol}{std_mcc:0.4f}"

                        mean_acc = np.nanmean(performance_info.overall_accuracy)
                        std_acc = np.nanstd(performance_info.overall_accuracy)
                        acc_string = f"{mean_acc:0.4f}{pm_symbol}{std_acc:0.4f}"

                        mean_auc = np.nanmean(performance_info.overall_auc)
                        std_auc = np.nanstd(performance_info.overall_auc)
                        auc_string = f"{mean_auc:0.4f}{pm_symbol}{std_auc:0.4f}"

                        # build string to print (using LaTex codes for the different columns)
                        mrs_aus = f"\\Block{{12-1}}{{{mrs}}}"
                        model_aus = f"\\Block{{6-1}}{{\\rotnd{{{model}}}}}"
                        use_age_aus = f"\\Block{{3-1}}{{{str(use_age)}}}"
                        pre_t_aus = f'\\Block{{{2 if pre_training else 1}-1}}{{{str(("SimCLR" if pre_training else "/") )}}}'
                        pre_t_d_aus = (
                            f"\\Block{{1-1}}{{{pre_training_dataset.title()}}}"
                        )

                        mcc_aus = f"\\Block{{1-1}}<\\{'f'}ootnotesize>{{${mean_mcc:0.4f} \\pm {std_mcc:0.4f}$}}"
                        acc_aus = f"\\Block{{1-1}}<\\{'f'}ootnotesize>{{${mean_acc:0.4f} \\pm {std_acc:0.4f}$}}"
                        auc_aus = f"\\Block{{1-1}}<\\{'f'}ootnotesize>{{${mean_auc:0.4f} \\pm {std_auc:0.4f}$}}"

                        mrs_sequence_string = f"{mrs_aus if print_mrs else mr_sequence_string_space:{len(mr_sequence_string_space)}s}"
                        model_string = f"{model_aus if print_model else models_string_space:{len(models_string_space)}s}"
                        use_age_string = f"{use_age_aus if print_use_age else age_string_space:{len(age_string_space)}s}"
                        pre_t_string = f"{pre_t_aus if print_pre_t else pre_training_string_space:{len(pre_training_string_space)}s}"
                        pre_t_d_string = f"{pre_t_d_aus if print_pre_t_d else pre_training_dataset_string_space:{len(pre_training_dataset_string_space)}s}"
                        mcc_string = f"{mcc_aus if print_metric else metric_space:{len(metric_space)}s}"
                        acc_string = f"{acc_aus if print_metric else metric_space:{len(metric_space)}s}"
                        auc_string = f"{auc_aus if print_metric else metric_space:{len(metric_space)}s}"

                        # this is quite tricky, but bare with me :)
                        list_strings_to_plot.append(
                            "&".join(
                                [
                                    mrs_sequence_string,
                                    model_string,
                                    use_age_string,
                                    pre_t_string,
                                    pre_t_d_string,
                                    mcc_string,
                                    acc_string,
                                    auc_string,
                                ]
                            )
                            + f"\\\\"
                        )

                        (
                            print_mrs,
                            print_model,
                            print_use_age,
                            print_pre_t,
                            print_pre_t_d,
                            print_metric,
                        ) = (False, False, False, False, False, False)

                    else:
                        # build empty string since model is still training
                        not_available_string = "NaN"
                        for idc, class_name in enumerate(CLASSES):
                            class_wise_metric_strings = []
                            for metric, values in zip(
                                ["precision", "recall", "f1-score"],
                                [
                                    (not_available_string, not_available_string),
                                    (not_available_string, not_available_string),
                                    (not_available_string, not_available_string),
                                ],
                            ):
                                class_wise_metric_string = (
                                    f"{values[0]}{pm_symbol}{values[1]}"
                                )
                                class_wise_metric_strings.append(
                                    f"{class_wise_metric_string:{len(metric_space)}s}"
                                )

                            # build string to print
                            mrs_sequence_string = f"{mrs if print_mrs else mr_sequence_string_space:{len(mr_sequence_string_space)}s}"
                            model_string = f"{model if print_model else models_string_space:{len(models_string_space)}s}"
                            use_age_string = f"{str(use_age) if print_use_age else age_string_space:{len(age_string_space)}s}"
                            pre_t_string = f"{str(pre_training) if print_pre_t else pre_training_string_space:{len(pre_training_string_space)}s}"
                            pre_t_d_string = f"{pre_training_dataset if print_pre_t_d else pre_training_dataset_string_space:{len(pre_training_dataset_string_space)}s}"
                            mcc_string = f"{not_available_string if print_metric else metric_space:{len(metric_space)}s}"
                            acc_string = f"{not_available_string if print_metric else metric_space:{len(metric_space)}s}"
                            auc_string = f"{not_available_string if print_metric else metric_space:{len(metric_space)}s}"
                            class_name_string = f"{class_name:{len(class_space)}s}"

                            # this is quite tricky, but bare with me :)
                            list_strings_to_plot.append(
                                "".join(
                                    [
                                        mrs_sequence_string,
                                        model_string,
                                        use_age_string,
                                        pre_t_string,
                                        pre_t_d_string,
                                        mcc_string,
                                        acc_string,
                                        auc_string,
                                        class_name_string,
                                        " ".join(class_wise_metric_strings),
                                    ]
                                )
                                + f"\\\\"
                            )

                            (
                                print_mrs,
                                print_model,
                                print_use_age,
                                print_pre_t,
                                print_pre_t_d,
                                print_metric,
                            ) = (False, False, False, False, False, False)

for line in list_strings_to_plot:
    print(line)
