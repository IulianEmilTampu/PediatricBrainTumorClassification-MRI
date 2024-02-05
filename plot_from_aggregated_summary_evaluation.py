# %%
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


# %% UTILITIES
def make_summary_plot(df, plot_settings, df_ensemble=None):
    # check if the data is available for all the box plots
    expected_plots = [
        (m, p) for m in plot_settings["x_order"] for p in plot_settings["hue_order"]
    ]
    # check if we have the data for these combinations
    expected_plots_flags = []
    for c in expected_plots:
        # filter data based on the model type and the pre-training. If it exists, flag as 1, else as 0
        if (
            len(
                df.loc[
                    (df[plot_settings["x_variable"]] == c[0])
                    & (df[plot_settings["hue_variable"]] == c[1])
                ]
            )
            == 0
        ):
            expected_plots_flags.append(0)
        else:
            expected_plots_flags.append(1)

    box_plot = sns.boxplot(
        x=plot_settings["x_variable"],
        y=plot_settings["metric_to_plot"],
        hue=plot_settings["hue_variable"],
        data=df,
        palette="Set3",
        hue_order=plot_settings["hue_order"],
        order=plot_settings["x_order"],
        ax=plot_settings["axis"] if plot_settings["axis"] else None,
    )
    # ### refine plot

    # ## title
    box_plot.set_title(
        plot_settings["figure_title"],
        fontsize=plot_settings["title_font_size"],
    )

    # ## fix axis
    # format y axis
    box_plot.set_ylim([plot_settings["ymin"], plot_settings["ymax"]])

    ylabels = [f"{x:,.2f}" for x in box_plot.get_yticks()]
    box_plot.set_yticklabels(ylabels, fontsize=plot_settings["y_axis_tick_font_size"])
    box_plot.set_ylabel(
        plot_settings["metric_to_plot_name"],
        fontsize=plot_settings["y_axis_label_font_size"],
    )

    # format x axis
    box_plot.set_xlabel("")
    box_plot.set_xticklabels(
        box_plot.get_xticklabels(),
        rotation=plot_settings["x_axis_tick_rotation"],
        fontsize=plot_settings["x_axis_tick_font_size"],
    )

    # ## add pattern to boxplots and legend
    hatches = []
    hatches.extend(
        plot_settings["available_hatches"][0 : len(plot_settings["hue_order"])]
        * len(plot_settings["x_order"])
    )
    legend_colors, legend_hatch = [], []

    # fix the boxplot patches to match the expected boxplots (and take care of those that are not present)
    list_of_boxplot_patches = [
        patch for patch in box_plot.patches if type(patch) == mpl.patches.PathPatch
    ][::-1]
    list_of_boxplot_patches_corrected = []
    for b in expected_plots_flags:
        if b == 1:
            list_of_boxplot_patches_corrected.append(list_of_boxplot_patches.pop())
        else:
            list_of_boxplot_patches_corrected.append(0)

    for i, patch in enumerate(list_of_boxplot_patches_corrected):
        if patch != 0:
            # Boxes from left to right
            hatch = hatches[i]
            patch.set_hatch(hatch)
            patch.set_edgecolor("k")
            patch.set_linewidth(1)
            # save color and hatch for the legend
            legend_colors.append(patch.get_facecolor())
            legend_hatch.append(hatch)

    # ## add ensemble
    if plot_settings["show_ensemble"]:
        # fix the df_ensemble to have the mean value of the metric in a column called value
        df_ensemble = (
            df_ensemble.groupby(
                [plot_settings["x_variable"], plot_settings["hue_variable"]]
            )
            .apply(lambda x: np.mean(x[plot_settings["metric_to_plot"]]))
            .rename("ensemble_value")
            .reset_index()
        )

        add_ensemble_values(
            df_ensemble=df_ensemble,
            plot_settings=plot_settings,
            expected_plots_flags=expected_plots_flags,
            ax=box_plot,
        )

    # ## fix legend
    if plot_settings["plot_legend"]:
        legend_labels = [
            f"{plot_settings['hue_variable'].title()} = {l.get_text()}"
            for idx, l in enumerate(box_plot.legend_.texts)
        ]
        # make patches (based on hue)
        legend_patch_list = [
            mpl.patches.Patch(fill=True, label=l, hatch=h, facecolor=c, edgecolor="k")
            for l, c, h in zip(legend_labels, legend_colors, legend_hatch)
        ]
        # if the Ensemble value is showed, add this to the legend
        if plot_settings["show_ensemble"]:
            legend_patch_list.append(
                mlines.Line2D(
                    [],
                    [],
                    color="k",
                    marker="X",
                    linestyle="None",
                    markersize=20,
                    label="Ensemble",
                )
            )
        # remake legend
        box_plot.legend(
            handles=legend_patch_list,
            loc="best",
            handleheight=4,
            handlelength=6,
            labelspacing=1,
            fontsize=plot_settings["legend_font_size"],
            # bbox_to_anchor=(1.01, 1),
        )
    else:
        box_plot.get_legend().remove()


def add_ensemble_values(
    df_ensemble,
    ax,
    plot_settings,
    expected_plots_flags,
    show_ensemble_value: bool = True,
    numeric_value_settings: dict = None,
    available_markers: list = ["s", "v", "^", "p", "X", "8", "*"],
):
    """
    df_ensemble is a pandas dataframe that has the ensemble value to plot for each of the x-hue combinations.
    """
    # get the location of the box plots in the axis
    lines = ax.get_lines()
    boxes = [
        c for c in ax.get_children() if type(c).__name__ == "PathPatch"
    ]  # this is the number of elements in the x axes
    lines_per_box = int(
        len(lines) / len(boxes)
    )  # this is the number of lines for each box (if 3 box plots -> 6 lines (2 lines for each box-plot))
    median_lines = lines[4 : len(lines) : lines_per_box]

    list_of_boxplot_patches_corrected = []
    list_of_medians_corrected = []
    for b in expected_plots_flags:
        if b == 1:
            list_of_boxplot_patches_corrected.append(boxes.pop(0))
            list_of_medians_corrected.append(median_lines.pop(0))
        else:
            list_of_boxplot_patches_corrected.append(0)
            list_of_medians_corrected.append(0)

    # create tuples of (x, hue) used to filter the dataframe
    x_hue_filter = [
        (x, hue) for x in plot_settings["x_order"] for hue in plot_settings["hue_order"]
    ]

    # loop through the different boxes using the x_order and hue_order information for filtering the df.
    # Also take care of if a box plot is missing
    for i, box in enumerate(list_of_boxplot_patches_corrected):
        if box != 0:
            # get x location of the box
            x = list_of_medians_corrected[i].get_data()[0].mean()
            # get y location based on the value of the ensemble. Take the mean over all the ensemble values
            y = df_ensemble.loc[
                (df_ensemble[plot_settings["x_variable"]] == x_hue_filter[i][0])
                & (df_ensemble[plot_settings["hue_variable"]] == x_hue_filter[i][1])
            ].ensemble_value.values[0]

            ax.scatter(x, y, marker="X", color="k", edgecolors="white", s=500, zorder=5)

            if plot_settings["show_ensemble_value"]:
                # add value to the parcker in a box
                text_box = dict(boxstyle="round", facecolor="white", alpha=0.7)
                if plot_settings["numeric_value_settings"]:
                    ax.text(
                        x=x,
                        y=y - plot_settings["ensemble_numeric_value_space"],
                        s=f"{y:0.2f}",
                        # s=f"{model_version:s}",
                        zorder=6,
                        bbox=text_box,
                        **plot_settings["numeric_value_settings"],
                    )
                else:
                    # use default
                    ax.text(
                        x=x,
                        y=y - plot_settings["ensemble_numeric_value_space"],
                        s=f"{y:0.2f}",
                        # s=f"{model_version:s}",
                        zorder=6,
                        c="k",
                        fontsize=25,
                        bbox=text_box,
                        path_effects=[
                            pe.withStroke(linewidth=2, foreground="ghostwhite")
                        ],
                    )


# add string to dataframe
def make_summary_string(x):
    # get all the information needed
    mrs = x.mr_sequence
    model = x.model_version
    prt = x.pretraining_type_str

    return f"{mrs}\n{model}\n{prt}"


# %% IMPORT SUMMARY FILE
SUMMARY_FILE_PATH = "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/evaluation_results/Evaluation_20240205/summary_evaluation_aggregated.csv"
SAVE_PATH = pathlib.Path(
    os.path.join(os.path.dirname(SUMMARY_FILE_PATH), "Summary_plots")
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


# %% SUBJECT_LEVEL SUMMARY

SAVE_FIGURE = True

# define the order in the plot
mr_sequence_order = ["T1", "T2", "ADC"]
model_version_order = ["ResNet50", "ViT_b_16"]
pretraining_version_order = ["ImageNet", "SimCLR_TCGA", "SimCLR_CBTN"]

# the x order to plot
x_order = [
    f"{mrs}\n{model}\n{prt}"
    for mrs in mr_sequence_order
    for model in model_version_order
    for prt in pretraining_version_order
]


# add summary string to dataframe
ORIGINAL_DF_SUBJECT_LEVEL["summary_string"] = ORIGINAL_DF_SUBJECT_LEVEL.apply(
    lambda x: make_summary_string(x), axis=1
)
ORIGINAL_DF_SUBJECT_LEVEL_ENSEMBLE["summary_string"] = (
    ORIGINAL_DF_SUBJECT_LEVEL_ENSEMBLE.apply(lambda x: make_summary_string(x), axis=1)
)

# plot and save for all the metrics
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


for metric, metric_specs in metric_and_text.items():
    # refine plot settings
    plot_settings = {
        "metric_to_plot": metric,
        "metric_to_plot_name": metric_specs["metric_name"],
        "x_variable": "summary_string",
        "x_order": x_order,
        "hue_variable": "use_age",
        "hue_order": [False, True],
        "figure_title": "Subject-wise performance",
        "show_ensemble": True,
        "show_ensemble_value": True,
        "tick_font_size": 12,
        "title_font_size": 20,
        "y_axis_label_font_size": 20,
        "y_axis_tick_font_size": 20,
        "x_axis_tick_font_size": 12,
        "x_axis_tick_rotation": 45,
        "legend_font_size": 12,
        "available_hatches": [
            "\\",
            "xx",
            "/",
            "\\",
            "|",
            "-",
            ".",
        ],
        "ymin": metric_specs["metric_range"][0] - 0.1,
        "ymax": metric_specs["metric_range"][1],
        "plot_legend": True,
        "axis": None,
        "numeric_value_settings": {"fontsize": 12, "rotation": -45},
        "ensemble_numeric_value_space": 0.09,
    }

    # plot
    fig = plt.figure(figsize=(22, 10))
    make_summary_plot(
        ORIGINAL_DF_SUBJECT_LEVEL,
        plot_settings,
        df_ensemble=ORIGINAL_DF_SUBJECT_LEVEL_ENSEMBLE,
    )

    if SAVE_FIGURE:
        file_name = f"Summary_classification_performance_subjects_wise_{metric.title()}"
        fig.savefig(
            os.path.join(SAVE_PATH, file_name + ".pdf"), bbox_inches="tight", dpi=100
        )
        fig.savefig(
            os.path.join(SAVE_PATH, file_name + ".png"), bbox_inches="tight", dpi=100
        )
        plt.close(fig)
    else:
        plt.show()

# %% SLICE-LEVEL SUMMARY

SAVE_FIGURE = True
# define the order in the plot
mr_sequence_order = ["T1", "T2", "ADC"]
model_version_order = ["ResNet50", "ViT_b_16"]
pretraining_version_order = ["ImageNet", "SimCLR_TCGA", "SimCLR_CBTN"]

# the x order to plot
x_order = [
    f"{mrs}\n{model}\n{prt}"
    for mrs in mr_sequence_order
    for model in model_version_order
    for prt in pretraining_version_order
]

# add summary string to the dataframe
ORIGINAL_DF_SLICE_LEVEL["summary_string"] = ORIGINAL_DF_SLICE_LEVEL.apply(
    lambda x: make_summary_string(x), axis=1
)
ORIGINAL_DF_SLICE_LEVEL_ENSEMBLE["summary_string"] = (
    ORIGINAL_DF_SLICE_LEVEL_ENSEMBLE.apply(lambda x: make_summary_string(x), axis=1)
)

# plot and save for all the metrics
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

for metric, metric_specs in metric_and_text.items():
    # refine plot settings
    plot_settings = {
        "metric_to_plot": metric,
        "metric_to_plot_name": metric_specs["metric_name"],
        "x_variable": "summary_string",
        "x_order": x_order,
        "hue_variable": "use_age",
        "hue_order": [False, True],
        "figure_title": "Subject-wise performance",
        "show_ensemble": True,
        "show_ensemble_value": True,
        "tick_font_size": 12,
        "title_font_size": 20,
        "y_axis_label_font_size": 20,
        "y_axis_tick_font_size": 20,
        "x_axis_tick_font_size": 12,
        "x_axis_tick_rotation": 45,
        "legend_font_size": 12,
        "available_hatches": [
            "\\",
            "xx",
            "/",
            "\\",
            "|",
            "-",
            ".",
        ],
        "ymin": metric_specs["metric_range"][0] - 0.1,
        "ymax": metric_specs["metric_range"][1],
        "plot_legend": True,
        "axis": None,
        "numeric_value_settings": {"fontsize": 12, "rotation": -45},
        "ensemble_numeric_value_space": 0.09,
    }

    # plot
    fig = plt.figure(figsize=(22, 10))
    make_summary_plot(
        ORIGINAL_DF_SLICE_LEVEL,
        plot_settings,
        df_ensemble=ORIGINAL_DF_SLICE_LEVEL_ENSEMBLE,
    )

    if SAVE_FIGURE:
        file_name = f"Summary_classification_performance_slice_wise_{metric.title()}"
        fig.savefig(
            os.path.join(SAVE_PATH, file_name + ".pdf"), bbox_inches="tight", dpi=100
        )
        fig.savefig(
            os.path.join(SAVE_PATH, file_name + ".png"), bbox_inches="tight", dpi=100
        )
        plt.close(fig)
    else:
        plt.show()


# %% WORK ON THE DETAILED EVALUATION
SUMMARY_FILE_PATH = "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/validation_results/Evaluation_20240117/detailed_evaluation_aggregated.csv"
SAVE_PATH = pathlib.Path(
    os.path.join(os.path.dirname(SUMMARY_FILE_PATH), "Summary_plots")
)
SAVE_PATH.mkdir(parents=True, exist_ok=True)
DETAILED_DF_ORIGINAL = pd.read_csv(SUMMARY_FILE_PATH, low_memory=False)

# use model with 0.5 fine tuning
DETAILED_DF_ORIGINAL = DETAILED_DF_ORIGINAL.loc[DETAILED_DF_ORIGINAL.fine_tuning == 0.5]

# use only the values from the evaluation set
DETAILED_DF_ORIGINAL = DETAILED_DF_ORIGINAL.loc[
    DETAILED_DF_ORIGINAL.evaluation_set == EVALUATION_SET
]
DETAILED_DF_ORIGINAL = DETAILED_DF_ORIGINAL.loc[
    DETAILED_DF_ORIGINAL.nbr_classes == NBR_CLASSES
]
# make an easy to use string for the pretraining type
DETAILED_DF_ORIGINAL = DETAILED_DF_ORIGINAL.assign(
    pretraining_type_str=DETAILED_DF_ORIGINAL.apply(
        lambda x: make_pretraining_type_string(x), axis=1
    )
)

# %% THIS IS GOING TO BE MAJECTIS!
"""
Here we are getting the performance information as a variable of modality combination (1 to 3), percentage of models used for ensemble, and relative position of the slices 
used for the subject ensemble.

Steps:
1 - Define which MR sequences are going to be used, along with the type of pretraining and if age is used.
2 - Randomly select models from the pool of models trained.
3 - For the overlapping subjects:
    3.1 - Get the slices in the specified range
    3.2 - average teh softmax scores for the classes for all the models and modalities
4 - Predict subject class and compute overall performance
5 - Save all into a dataframe for easy plotting
6 - repeat from 2 to 5 N times to get a boxplot.
"""
from itertools import chain, combinations
import evaluation_utilities


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def get_random_repetiton_fold_pair(
    max_nbr_repetition, max_nbr_folds, nbr_pairs_to_create, starting_from: int = 1
):
    seen = set()
    while len(seen) < nbr_pairs_to_create:
        # create random pair
        x, y = np.random.randint(starting_from, max_nbr_repetition), np.random.randint(
            starting_from, max_nbr_folds
        )
        # add the pair if it is not duplicate
        if not (x, y) in seen:
            seen.add((x, y))
    return list(seen)


def replace_all(string, dict_for_replacement: dict = {" ": "", "[": "", "]": ""}):
    for r, s in dict_for_replacement.items():
        string = string.replace(r, s)
    return string


NUMBER_ORANDOM_SAMPLINGS = 1
# PERCENTAGES_MODEL_POPULATION = [5, 25, 50, 75, 100]
# RLP_RANGES = [[20, 80], [30, 70], [40, 60], [48, 52]]

PERCENTAGES_MODEL_POPULATION = [5]
RLP_RANGES = [[20, 80]]

MR_SEQUENCES_AVAILABLE = list(pd.unique(DETAILED_DF_ORIGINAL.mr_sequence))
MR_SEQUENCES_COMBINATIONS = list(powerset(MR_SEQUENCES_AVAILABLE))[1::]
MR_SEQUENCES_COMBINATIONS = MR_SEQUENCES_COMBINATIONS[3:4]

# other model selection criterias (these can be used to specify different model version for the different modalities and so on. For now we keep it simple)
MODEL_VERSIONS = [
    ["ResNet50"] * len(combination) for combination in MR_SEQUENCES_COMBINATIONS
]
PRE_TRAINING_VERSIONS = ["ImageNet"] * len(MR_SEQUENCES_COMBINATIONS)
USE_AGE = [False] * len(MR_SEQUENCES_COMBINATIONS)

# initiate where to save the information
SUMMARY_COMPUTATION = []

# start the looping
for indx_n, n in enumerate(range(NUMBER_ORANDOM_SAMPLINGS)):
    for (
        mr_sequence_combination,
        model_versions,
        pre_training_versions,
        use_age_versions,
    ) in zip(MR_SEQUENCES_COMBINATIONS, MODEL_VERSIONS, PRE_TRAINING_VERSIONS, USE_AGE):
        # get list of overlapping subjects
        unique_subjects_per_modality = [
            list(
                pd.unique(
                    DETAILED_DF_ORIGINAL.loc[
                        DETAILED_DF_ORIGINAL.mr_sequence == mr_sequence
                    ]["subject_IDs"]
                )
            )
            for mr_sequence in mr_sequence_combination
        ]
        unique_overlapping_subjects = list(
            set.intersection(*[set(list_) for list_ in unique_subjects_per_modality])
        )
        # for each combination of mr sequence, model version and model selection critaria, get indexes from the pool of models
        for mr_seq, model_version, pre_training, use_age in zip(
            mr_sequence_combination,
            model_versions,
            pre_training_versions,
            use_age_versions,
        ):
            for percentage_of_models in PERCENTAGES_MODEL_POPULATION:
                # by default each model has 10 repetitions with 5 folds each. Randomly create tuples of repetition index fold index to
                # select as many models as the percentage_of_models
                nbr_models = int(50 * percentage_of_models / 100)
                random_repetition_fold_indexes = get_random_repetiton_fold_pair(
                    max_nbr_repetition=10 + 1,
                    max_nbr_folds=5 + 1,
                    nbr_pairs_to_create=nbr_models,
                )

                # for the unique overlapping subjects, get the subject softmax score obtained from the slices in the specified range.
                for slice_range in RLP_RANGES:
                    # this is the difficult part since we are filtering the datarfame with many conditions
                    for indexes in random_repetition_fold_indexes:
                        values = DETAILED_DF_ORIGINAL.loc[
                            (DETAILED_DF_ORIGINAL.mr_sequence == mr_seq)
                            & (DETAILED_DF_ORIGINAL.model_version == model_version)
                            & (
                                DETAILED_DF_ORIGINAL.pretraining_type_str
                                == pre_training
                            )
                            & (DETAILED_DF_ORIGINAL.use_age == use_age)
                            # subject ID filter
                            & (
                                DETAILED_DF_ORIGINAL.subject_IDs.isin(
                                    unique_overlapping_subjects
                                )
                            )
                            &
                            # repetition filter
                            (DETAILED_DF_ORIGINAL.repetition == indexes[0])
                            &
                            # filter based on the relative position
                            (
                                DETAILED_DF_ORIGINAL.tumor_relative_position
                                >= slice_range[0]
                            )
                            & (
                                DETAILED_DF_ORIGINAL.tumor_relative_position
                                <= slice_range[1]
                            )
                        ].loc[
                            :,
                            [
                                f"pred_fold_{indexes[1]}",
                                "subject_IDs",
                                "one_hot_encodig",
                            ],
                        ]
                        # for some reason the values are strings, so convert to floats
                        # the softmax values
                        string_values = list(values[f"pred_fold_{indexes[1]}"])
                        float_values = [
                            np.array(replace_all(v).split(","), dtype=np.float32)
                            for v in string_values
                        ]
                        values[f"pred_fold_{indexes[1]}"] = float_values

                        # the one hot encoding values
                        string_values = list(values["one_hot_encodig"])
                        float_values = [
                            np.array(replace_all(v).split(","), dtype=np.float32)
                            for v in string_values
                        ]
                        values[f"one_hot_encodig"] = float_values

                        # take the mean softmax across the slices for each subject separately
                        values = values.groupby("subject_IDs").mean().reset_index()
                        # add information about the model settings, percentage of models and slice range and save.
                        values["random_sampling_index"] = [indx_n] * len(values)
                        values["percentage_models"] = [percentage_of_models] * len(
                            values
                        )
                        values["slice_range"] = [slice_range] * len(values)
                        values["mr_sequence_combination"] = [
                            "_".join(mr_sequence_combination)
                        ] * len(values)
                        values["mr_sequence"] = [mr_seq] * len(values)
                        values["model_version"] = [model_version] * len(values)
                        values["pre_training"] = [pre_training] * len(values)
                        values["use_age"] = [use_age] * len(values)
                        # save
                        print(mr_seq)
                        SUMMARY_COMPUTATION.append(values)


# %% PLOT THE PER MODEL (with and withouth age), PER-FOLD AND PER CLASS ENTROPY -
# THIS IS A PER-SUBJECT EVALUATION SINCE WE ARE
# STILL LOOKING AT THE VALIDATION SET, THUS WE CAN NOT PERFORM A PER-SLICE ENTROPY EVALUATION (WE DON'T HAVE ON A PER-MODEL BASIS)
# MULTIPLE PREDICTIONS OF THE SAME SLIDE (VALIDATION SUBJECT DOES NOT APPEAR AS VALIDATION IN MULTIPLE FOLDS). THIS IS NOT THE CASE FOR THE
# TEST SET, THUS THIS EVALUATION CAN BE PEROFRMED WHEN HAVING THE TEST VALUES.

# filter dataset for plotting. Here get for each model version, get for each fold get all the entropy values from all the subjects
entropy_plot_df = deepcopy(DF)


# make ausiliary columns to help plotting
def make_pretraining_type_string(x):
    if not x.pretraining:
        return "ImageNet"
    else:
        return f"SimCLR_{str(x.pretraining_dataset).upper()}"


def make_model_version_string(x):
    if x.use_age:
        if x.age_encoder_MLP_nodes == 0:
            return f"{x.model_version} (i+a) SAE"
        elif x.age_encoder_MLP_nodes == 3:
            return f"{x.model_version} (i+a) LAE"
    else:
        return x.model_version


entropy_plot_df["pretraining_type_str"] = entropy_plot_df.apply(
    lambda x: make_pretraining_type_string(x), axis=1
)
entropy_plot_df["model_version"] = entropy_plot_df.apply(
    lambda x: make_model_version_string(x), axis=1
)


# group by the type of model version, pretraining type and subject_ID and take the mean over the folds of the subject entropies
def average_subject_entropy(x):
    # get the  fold-wise entropy for this subject (a litle bit triki since the values are strings. Drop those that are [None], and
    # convert the first value to an np array (all the values for this subject and fold are the same.)).
    per_fold_subject_entropy = [
        np.fromstring(x[col].values[0].strip("[]"), count=3, sep=",")
        for col in x.columns
        if all(["subject_entropy_pred_fold_" in col, x[col].values[0] != "[None]"])
    ]
    # take mean over the classes for all the folds
    return np.array(per_fold_subject_entropy).mean(axis=0)


entropy_plot_df = (
    entropy_plot_df.groupby(
        ["model_version", "pretraining_type_str", "subject_IDs"], group_keys=False
    )
    .apply(lambda x: average_subject_entropy(x))
    .reset_index()
)
entropy_plot_df = entropy_plot_df.rename(columns={0: "subject_entropy_over_the_folds"})


# split the subject_entropy_over_the_folds to get one columns for each class
def split_to_classes(x):
    for c in range(x.subject_entropy_over_the_folds.shape[-1]):
        x[f"subject_entropy_over_the_folds_class_{c+1}"] = (
            x.subject_entropy_over_the_folds[c]
        )

    return x


entropy_plot_df = entropy_plot_df.apply(lambda x: split_to_classes(x), axis=1)

# %%
sns.boxplot(
    x="model_version",
    y="subject_entropy_over_the_folds_class_3",
    hue="pretraining_type_str",
    data=entropy_plot_df,
)

# %% PER-RELATIVE POSITION ENTROPY COMPUTED OVER ALL THE MODELS WITH COLORCODED DOTS FOR THE DIFFERENT MODEL VERSIONS


# %% TEST
# Need to fix the fact that some boxplots might be missing, thus need to jump.
# Used the hue_pretraining_type_order and x_model_version_order
def add_ensemble_values(
    slice_level_ensemble_df,
    ax,
    x_order,
    hue_order,
    metric_to_plot,
    show_ensemble_value: bool = True,
    numeric_value_settings: dict = None,
    list_of_boxplot_patches_corrected: list = None,
):
    lines = ax.get_lines()
    boxes = [
        c for c in ax.get_children() if type(c).__name__ == "PathPatch"
    ]  # this is the number of elements in the x axes (model type in this case)
    lines_per_box = int(
        len(lines) / len(boxes)
    )  # this is the number of lines for each box (if 3 box plots -> 6 lines (2 lines for each box-plot))

    # define markers to use
    available_markers = ["s", "v", "^", "p", "X", "8", "*"]
    # index cycle. This is a little bit tricky since the boxplot order follows the hue order and then the x order.
    index_hue_cycle = list(range(len(hue_order))) * len(x_order)
    index_x_cycle = []
    [index_x_cycle.extend([i] * len(hue_order)) for i in range(len(x_order))]

    # loop through the different boxes using the x_order and hue_order information for filtering the df
    for idx, (h_idx, x_idx, median) in enumerate(
        zip(index_hue_cycle, index_x_cycle, lines[4 : len(lines) : lines_per_box])
    ):
        pretraining_type = hue_order[h_idx]
        model_version = x_order[x_idx]
        # get x location of the box
        x = median.get_data()[0].mean()
        # get y location based on the value of the ensemble. Take the mean over all the ensemble values
        y = np.mean(
            slice_level_ensemble_df.loc[
                (slice_level_ensemble_df.pretraining_type_str == pretraining_type)
                & (slice_level_ensemble_df.model_version == model_version)
            ][metric_to_plot]
        )
        ax.scatter(x, y, marker="X", color="k", edgecolors="white", s=500, zorder=5)
        if show_ensemble_value:
            # add value to the parcker in a box
            text_box = dict(boxstyle="round", facecolor="white", alpha=0.7)
            if numeric_value_settings:
                ax.text(
                    x=x,
                    y=y - 0.05,
                    s=f"{y:0.4f}",
                    # s=f"{model_version:s}",
                    zorder=6,
                    bbox=text_box,
                    **numeric_value_settings,
                )
            else:
                # use default
                ax.text(
                    x=x,
                    y=y - 0.05,
                    s=f"{y:0.4f}",
                    # s=f"{model_version:s}",
                    zorder=6,
                    c="k",
                    fontsize=25,
                    bbox=text_box,
                    path_effects=[pe.withStroke(linewidth=2, foreground="ghostwhite")],
                )


expected_plots = [
    (m, p) for m in x_model_version_order for p in hue_pretraining_type_order
]

# check if we have the data for these combinations
expected_plots_flags = []
for c in expected_plots:
    # filter data based on the model type and the pre-training. If it exists, flag as 1, else as 0
    if (
        len(
            slice_level_df.loc[
                (slice_level_df.model_version == c[0])
                & (slice_level_df.pretraining_type_str == c[1])
            ]
        )
        == 0
    ):
        expected_plots_flags.append(0)
    else:
        expected_plots_flags.append(1)

fig, box_plot = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

box_plot = sns.boxplot(
    x="model_version",
    y=metric_to_plot,
    hue="pretraining_type_str",
    data=slice_level_df,
    palette="Set3",
    hue_order=hue_pretraining_type_order,
    order=x_model_version_order,
)

# ## fix title and axis
box_plot.set_title(
    f"Slice-wise classification performance ({MR_SEQUENCE})",
    fontsize=title_font_size,
)
box_plot.tick_params(labelsize=tick_font_size)
box_plot.set_ylabel(
    f"{metric_name_for_plot}",
    fontsize=label_font_size,
)
# format y tick labels
ylabels = [f"{x:,.2f}" for x in box_plot.get_yticks()]
box_plot.set_yticklabels(ylabels)

box_plot.set_xlabel("")
box_plot.set_xticklabels(box_plot.get_xticklabels(), rotation=0)

# ## add pattern to boxplots and legend
available_hatches = [
    "\\",
    "xx",
    "/",
    "\\",
    "|",
    "-",
    ".",
]
hatches = []
# the order of the pathces in not as plotted. The pathes with the same hue value are consecutive in the list.
[
    hatches.extend(available_hatches[0 : len(hue_pretraining_type_order)])
    for i in range(len(x_model_version_order))
]
colors, legend_hatch = [], []

# # fix the boxplot patches to match the expected boxplots (and take care of those that are not present)
# list_of_boxplot_patches = [patch for patch in box_plot.patches if type(patch) == mpl.patches.PathPatch][::-1]
# list_of_boxplot_patches_corrected = []
# for b in expected_plots_flags:
#     if b == 1:
#         list_of_boxplot_patches_corrected.append(list_of_boxplot_patches.pop())
#     else:
#         list_of_boxplot_patches_corrected.append(0)


# for i, patch in enumerate(
#     list_of_boxplot_patches_corrected
# ):
#     if patch != 0:
#         # Boxes from left to right
#         hatch = hatches[i]
#         patch.set_hatch(hatch)
#         patch.set_edgecolor("k")
#         patch.set_linewidth(1)
#         colors.append(patch.get_facecolor())
#         legend_hatch.append(hatch)

# # ## add ensemble
# add_ensemble_values(
#     slice_level_ensemble_df,
#     box_plot,
#     x_model_version_order,
#     hue_pretraining_type_order,
#     metric_to_plot,
#     list_of_boxplot_patches_corrected,
# )

# ## fix legend
legend_labels = [f"{l.get_text()}" for l in box_plot.legend_.texts]
legend_colors = colors[
    :: len(colors) // len(hue_pretraining_type_order) if len(colors) > 1 else 1
]
legend_hatch = legend_hatch[
    :: (
        len(legend_hatch) // len(hue_pretraining_type_order)
        if len(legend_hatch) > 1
        else 1
    )
]

# make patches
legend_patch_list = [
    mpl.patches.Patch(fill=True, label=l, hatch=h, facecolor=c, edgecolor="k")
    for l, c, h in zip(legend_labels, legend_colors, legend_hatch)
]
# add ensemble marker
legend_patch_list.append(
    mlines.Line2D(
        [],
        [],
        color="k",
        marker="X",
        linestyle="None",
        markersize=20,
        label="Ensemble",
    )
)

# remake legend
box_plot.legend(
    handles=legend_patch_list,
    loc="best",
    handleheight=4,
    handlelength=6,
    labelspacing=1,
    fontsize=legend_font_size,
    bbox_to_anchor=(1.01, 1),
)

# ## final touches
box_plot.yaxis.grid(True, zorder=-3)  # Hide the horizontal gridlines
box_plot.xaxis.grid(False)  # Show the vertical gridlines
box_plot.figure.tight_layout()
