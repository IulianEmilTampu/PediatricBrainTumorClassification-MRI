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
def add_median_labels(ax, precision=".1f"):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == "PathPatch"]
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4 : len(lines) : lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(
            x,
            y,
            f"{value:{precision}}",
            ha="center",
            va="center",
            fontweight="bold",
            color="black",
        )
        # create median-colored border around white text for contrast
        # text.set_path_effects([
        #     path_effects.Stroke(linewidth=2, foreground=median.get_color()),
        #     path_effects.Normal(),
        # ])
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=2, foreground="white"),
                path_effects.Normal(),
            ]
        )


def add_ensemble_values(
    slice_level_ensemble_df,
    ax,
    x_order,
    hue_order,
    metric_to_plot,
    show_ensemble_value: bool = True,
    numeric_value_settings: dict = None,
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


# # # shaddow in the background
def add_shadow_between_hues(ax, y_min, y_max, alpha=0.05, zorder=30, color="black"):
    # get hue region positions
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == "PathPatch"]
    lines_per_box = int(len(lines) / len(boxes))

    # get the number of boxes per hue region
    nbr_boxes_per_hue = int(len(boxes) / len(ax.get_xticks()))

    # build coordinate regions where to add shadow
    # starting from the 0th or 1st hue gerion
    start_hue_region = 0
    # take the initial coordinate of the first box of the region and
    # the last of the last box of the region
    # get coordinatex for all boxes in order
    x_boxes_coordinates = []
    for idx, median in enumerate(lines[4 : len(lines) : lines_per_box]):
        # get x location of the box
        x_boxes_coordinates.append(median.get_data()[0])

    # get hue region coordinate
    hue_region_coordinatex = []
    for hue_region in range(len(ax.get_xticks())):
        idx_first_box = hue_region * nbr_boxes_per_hue
        idx_last_box = idx_first_box + nbr_boxes_per_hue - 1
        hue_region_coordinatex.append(
            [
                x_boxes_coordinates[idx_first_box][0],
                x_boxes_coordinates[idx_last_box][-1],
            ]
        )

    # loop through the regions and color
    for c in range(start_hue_region, len(ax.get_xticks()), 2):
        x_min, x_max = hue_region_coordinatex[c][0], hue_region_coordinatex[c][-1]
        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                (x_max - x_min),
                (y_max - y_min),
                color=color,
                alpha=alpha,
                zorder=zorder,
            )
        )


# %% IMPORT SUMMARY FILE
SUMMARY_FILE_PATH = "/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/validation_results/Evaluation_20240117/summary_evaluation_aggregated.csv"
SAVE_PATH = pathlib.Path(
    os.path.join(os.path.dirname(SUMMARY_FILE_PATH), "Summary_plots")
)
SAVE_PATH.mkdir(parents=True, exist_ok=True)
ORIGINAL_DF = pd.read_csv(SUMMARY_FILE_PATH)

# use model with 0.5 fine tuning
ORIGINAL_DF = ORIGINAL_DF.loc[ORIGINAL_DF.fine_tuning == 0.5]
# use plot results for a 3-class classification problem
ORIGINAL_DF = ORIGINAL_DF.loc[ORIGINAL_DF.nbr_classes == 3]

# MR sequence to use
MR_SEQUENCE = "ADC"
# SET to look at
EVALUATION_SET = "test"
ORIGINAL_DF = ORIGINAL_DF.loc[ORIGINAL_DF.evaluation_set == EVALUATION_SET]
# number of classes
NBR_CLASSES = 3
ORIGINAL_DF = ORIGINAL_DF.loc[ORIGINAL_DF.nbr_classes == NBR_CLASSES]

# %% PLOT SLICE-WISE BOXPLOTS FOR THE DIFFERENT MDOEL VERSIONS USING AS GROUPING THE DIFFERENT PRE-TRAINING TYPES
# Filter the dataframe to only get the metrics for the different folds of the repetitions on a slide level

SAVE = False
USE_AGE = False
WHAT = "slice_wise_no_age"

# select mr sequence
DF = ORIGINAL_DF.loc[ORIGINAL_DF["mr_sequence"] == MR_SEQUENCE]
# filter based on slice_levell predictions
FILTER = [f"pred_fold_{i+1}" for i in range(10)]
slice_level_df = DF[DF["performance_over"].isin(FILTER)]
slice_level_df = slice_level_df.loc[slice_level_df.use_age == USE_AGE]

slice_level_ensemble_df = DF.loc[
    (DF.performance_over == "per_slice_ensemble") & (DF.use_age == USE_AGE)
]


# Create s string for the different types of pretraining
# pretraining==False -> ImageNet
# pretraining==True $ pretraining_dataset == tcga -> SimCLR_TCGA
# pretraining==True $ pretraining_dataset == cbtn -> SimCLR_CBTN
def make_pretraining_type_string(x):
    if not x.pretraining:
        return "ImageNet"
    else:
        return f"SimCLR_{str(x.pretraining_dataset).upper()}"


slice_level_df["pretraining_type_str"] = slice_level_df.apply(
    lambda x: make_pretraining_type_string(x), axis=1
)
slice_level_ensemble_df["pretraining_type_str"] = slice_level_ensemble_df.apply(
    lambda x: make_pretraining_type_string(x), axis=1
)

# start plotting
metric_and_text = {
    "overall_precision": "Precision [0,1]",
    "overall_recall": "Recall [0,1]",
    "overall_accuracy": "Accuracy [0,1]",
    "overall_f1-score": "F1-score [0,1]",
    "overall_auc": "AUC [0,1]",
    "matthews_correlation_coefficient": "Matthews correlation coefficient [-1,1]",
}

metric_and_text = {
    "matthews_correlation_coefficient": "Matthews correlation coefficient [-1,1]"
}


hue_pretraining_type_order = ["ImageNet", "SimCLR_TCGA", "SimCLR_CBTN"]
x_model_version_order = ["ResNet50", "ViT_b_16"]

tick_font_size = 17
title_font_size = 20
label_font_size = 20
legend_font_size = 12

for metric_to_plot, metric_name_for_plot in metric_and_text.items():
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
        hatches.extend([available_hatches[i]] * len(x_model_version_order))
        for i in range(len(hue_pretraining_type_order))
    ]
    colors, legend_hatch = [], []

    for i, patch in enumerate(
        [patch for patch in box_plot.patches if type(patch) == mpl.patches.PathPatch]
    ):
        # Boxes from left to right
        hatch = hatches[i]
        patch.set_hatch(hatch)
        patch.set_edgecolor("k")
        patch.set_linewidth(1)
        colors.append(patch.get_facecolor())
        legend_hatch.append(hatch)

    # ## add ensemble
    add_ensemble_values(
        slice_level_ensemble_df,
        box_plot,
        x_model_version_order,
        hue_pretraining_type_order,
        metric_to_plot,
    )

    # ## fix legend
    legend_labels = [f"{l.get_text()}" for l in box_plot.legend_.texts]
    legend_colors = colors[
        :: len(colors) // len(hue_pretraining_type_order) if len(colors) > 1 else 1
    ]
    legend_hatch = legend_hatch[
        :: len(legend_hatch) // len(hue_pretraining_type_order)
        if len(legend_hatch) > 1
        else 1
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
    if SAVE == True:
        file_name = (
            f"{MR_SEQUENCE}_Summary_classification_performance_{WHAT}_{metric_to_plot}"
        )
        fig.savefig(
            os.path.join(SAVE_PATH, file_name + ".pdf"), bbox_inches="tight", dpi=100
        )
        fig.savefig(
            os.path.join(SAVE_PATH, file_name + ".png"), bbox_inches="tight", dpi=100
        )
        plt.close(fig)
    else:
        plt.show()

# %% PLOT SUBJECT-WISE BOXPLOTS FOR THE DIFFERENT MDOEL VERSIONS USING AS GROUPING THE DIFFERENT PRE-TRAINING TYPES
# Filter the dataframe to only get the metrics for the different folds of the repetitions on a slide level

SAVE = False
USE_AGE = False
WHAT = "subject_wise_no_age"

# select mr sequence
DF = ORIGINAL_DF.loc[ORIGINAL_DF["mr_sequence"] == MR_SEQUENCE]
# filter based on subject_level predictions predictions
FILTER = [f"subject_ensemble_pred_fold_{i+1}" for i in range(10)]
slice_level_df = DF[DF["performance_over"].isin(FILTER)]
slice_level_df = slice_level_df.loc[slice_level_df.use_age == USE_AGE]

slice_level_ensemble_df = DF.loc[
    (DF.performance_over == "overall_subject_ensemble") & (DF.use_age == USE_AGE)
]

# Create s string for the different types of pretraining
# pretraining==False -> ImageNet
# pretraining==True $ pretraining_dataset == tcga -> SimCLR_TCGA
# pretraining==True $ pretraining_dataset == cbtn -> SimCLR_CBTN


def make_pretraining_type_string(x):
    if not x.pretraining:
        return "ImageNet"
    else:
        return f"SimCLR_{str(x.pretraining_dataset).upper()}"


slice_level_df["pretraining_type_str"] = slice_level_df.apply(
    lambda x: make_pretraining_type_string(x), axis=1
)
slice_level_ensemble_df["pretraining_type_str"] = slice_level_ensemble_df.apply(
    lambda x: make_pretraining_type_string(x), axis=1
)

# start plotting
metric_and_text = {
    "overall_precision": "Precision [0,1]",
    "overall_recall": "Recall [0,1]",
    "overall_accuracy": "Accuracy [0,1]",
    "overall_f1-score": "F1-score [0,1]",
    "overall_auc": "AUC [0,1]",
    "matthews_correlation_coefficient": "Matthews correlation coefficient [-1,1]",
}

metric_and_text = {
    "matthews_correlation_coefficient": "Matthews correlation coefficient [-1,1]"
}

hue_pretraining_type_order = ["ImageNet", "SimCLR_TCGA", "SimCLR_CBTN"]
x_model_version_order = ["ResNet50", "ViT_b_16"]

tick_font_size = 17
title_font_size = 20
label_font_size = 20
legend_font_size = 12

for metric_to_plot, metric_name_for_plot in metric_and_text.items():
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
        f"SUbject-wise classification performance ({MR_SEQUENCE})",
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
    hatches.extend(
        available_hatches[0 : len(hue_pretraining_type_order)]
        * len(x_model_version_order)
    )
    colors, legend_hatch = [], []

    for i, patch in enumerate(
        [patch for patch in box_plot.patches if type(patch) == mpl.patches.PathPatch]
    ):
        # Boxes from left to right
        hatch = hatches[i]
        patch.set_hatch(hatch)
        patch.set_edgecolor("k")
        patch.set_linewidth(1)
        colors.append(patch.get_facecolor())
        legend_hatch.append(hatch)

    # ## add ensemble
    add_ensemble_values(
        slice_level_ensemble_df,
        box_plot,
        x_model_version_order,
        hue_pretraining_type_order,
        metric_to_plot,
    )

    # ## fix legend
    legend_labels = [f"{l.get_text()}" for l in box_plot.legend_.texts]
    legend_colors = colors[:: len(colors) // len(hue_pretraining_type_order)]
    legend_hatch = legend_hatch[:: len(legend_hatch) // len(hue_pretraining_type_order)]

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
    if SAVE == True:
        file_name = f"Summary_classification_performance_{WHAT}_{metric_to_plot}"
        fig.savefig(
            os.path.join(SAVE_PATH, file_name + ".pdf"), bbox_inches="tight", dpi=100
        )
        fig.savefig(
            os.path.join(SAVE_PATH, file_name + ".png"), bbox_inches="tight", dpi=100
        )
        plt.close(fig)
    else:
        plt.show()

# %% PLOT SLICE-SIZE BOXPLOTS FOR THE DIFFERENT MDOEL VERSIONS USING AS GROUPING THE DIFFERENT PRE-TRAINING TYPES (with and without age)
# Filter the dataframe to only get the metrics for the different folds of the repetitions on a slide level

SAVE = False
WHAT = "slice_wise"

# select mr sequence
DF = ORIGINAL_DF.loc[ORIGINAL_DF["mr_sequence"] == MR_SEQUENCE]

FILTER = [f"pred_fold_{i+1}" for i in range(20)]
slice_level_df = DF[DF["performance_over"].isin(FILTER)]
slice_level_ensemble_df = DF.loc[(DF.performance_over == "per_slice_ensemble")]

# Create s string for the different types of pretraining
# pretraining==False -> ImageNet
# pretraining==True $ pretraining_dataset == tcga -> SimCLR_TCGA
# pretraining==True $ pretraining_dataset == cbtn -> SimCLR_CBTN

slice_level_df = slice_level_df.assign(
    pretraining_type_str=slice_level_df.apply(
        lambda x: make_pretraining_type_string(x), axis=1
    )
)

slice_level_ensemble_df = slice_level_ensemble_df.assign(
    pretraining_type_str=slice_level_ensemble_df.apply(
        lambda x: make_pretraining_type_string(x), axis=1
    )
)

slice_level_df = slice_level_df.assign(
    model_version=slice_level_df.apply(lambda x: make_model_version_string(x), axis=1)
)
slice_level_ensemble_df = slice_level_ensemble_df.assign(
    model_version=slice_level_ensemble_df.apply(
        lambda x: make_model_version_string(x), axis=1
    )
)

# start plotting
metric_and_text = {
    "overall_precision": "Precision [0,1]",
    "overall_recall": "Recall [0,1]",
    "overall_accuracy": "Accuracy [0,1]",
    "overall_f1-score": "F1-score [0,1]",
    "overall_auc": "AUC [0,1]",
    "matthews_correlation_coefficient": "Matthews correlation coefficient [-1,1]",
}

metric_and_text = {
    "matthews_correlation_coefficient": "Matthews correlation coefficient [-1,1]"
}

hue_pretraining_type_order = ["ImageNet", "SimCLR_TCGA", "SimCLR_CBTN"]
# x_model_version_order = [
#     "ResNet50",
#     "ResNet50 (i+a) SAE",
#     "ResNet50 (i+a) LAE",
#     "ViT_b_16",
#     "ViT_b_16 (i+a) SAE",
#     "ViT_b_16 (i+a) LAE",
# ]

x_model_version_order = [
    "ResNet50",
    "ResNet50 (i+a) SAE",
    "ViT_b_16",
    "ViT_b_16 (i+a) SAE",
]

tick_font_size = 17
title_font_size = 20
label_font_size = 20
legend_font_size = 12

for metric_to_plot, metric_name_for_plot in metric_and_text.items():
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
    box_plot.set_title(f"Classification performance", fontsize=title_font_size)
    box_plot.tick_params(labelsize=tick_font_size)
    box_plot.set_ylabel(
        f"{metric_name_for_plot}",
        fontsize=label_font_size,
    )
    # format y tick labels
    ylabels = [f"{x:,.2f}" for x in box_plot.get_yticks()]
    box_plot.set_yticklabels(ylabels)

    box_plot.set_xlabel("")
    box_plot.set_xticklabels(box_plot.get_xticklabels(), rotation=-15, ha="left")

    # ## add pattern to boxplots and legend
    available_hatches = [
        "\\",
        "xx",
        "/",
        "\\",
        "|",
        "-",
        ".",
        "oo",
        "\|",
        "/|",
        "-\ ",
        "-/",
    ]
    hatches = []
    hatches.extend(
        available_hatches[0 : len(hue_pretraining_type_order)]
        * len(x_model_version_order)
    )
    colors, legend_hatch = [], []

    for i, patch in enumerate(
        [patch for patch in box_plot.patches if type(patch) == mpl.patches.PathPatch]
    ):
        # Boxes from left to right
        hatch = hatches[i]
        patch.set_hatch(hatch)
        patch.set_edgecolor("k")
        patch.set_linewidth(1)
        colors.append(patch.get_facecolor())
        legend_hatch.append(hatch)

    # ## add ensemble
    add_ensemble_values(
        slice_level_ensemble_df,
        box_plot,
        x_model_version_order,
        hue_pretraining_type_order,
        metric_to_plot,
        numeric_value_settings={"fontsize": 12, "rotation": -45},
    )

    # ## fix legend
    legend_labels = [f"{l.get_text()}" for l in box_plot.legend_.texts]
    legend_colors = colors[:: len(colors) // len(hue_pretraining_type_order)]
    legend_hatch = legend_hatch[:: len(legend_hatch) // len(hue_pretraining_type_order)]

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
    if SAVE == True:
        file_name = f"Summary_classification_performance_{WHAT}_{metric_to_plot}"
        fig.savefig(
            os.path.join(SAVE_PATH, file_name + ".pdf"), bbox_inches="tight", dpi=100
        )
        fig.savefig(
            os.path.join(SAVE_PATH, file_name + ".png"), bbox_inches="tight", dpi=100
        )
        plt.close(fig)
    else:
        plt.show()

# %% PLOT SUBJECT-SIZE BOXPLOTS FOR THE DIFFERENT MDOEL VERSIONS USING AS GROUPING THE DIFFERENT PRE-TRAINING TYPES (with and without age)
# Filter the dataframe to only get the metrics for the different folds of the repetitions on a slide level

SAVE = False
WHAT = "subject_wise"

# select mr sequence
DF = ORIGINAL_DF.loc[ORIGINAL_DF["mr_sequence"] == MR_SEQUENCE]

FILTER = [f"subject_ensemble_pred_fold_{i+1}" for i in range(20)]
subject_level_df = DF[DF["performance_over"].isin(FILTER)]
subject_level_ensemble_df = DF.loc[(DF.performance_over == "overall_subject_ensemble")]

# Create s string for the different types of pretraining
# pretraining==False -> ImageNet
# pretraining==True $ pretraining_dataset == tcga -> SimCLR_TCGA
# pretraining==True $ pretraining_dataset == cbtn -> SimCLR_CBTN

subject_level_df = subject_level_df.assign(
    pretraining_type_str=subject_level_df.apply(
        lambda x: make_pretraining_type_string(x), axis=1
    )
)
subject_level_ensemble_df = subject_level_ensemble_df.assign(
    pretraining_type_str=subject_level_ensemble_df.apply(
        lambda x: make_pretraining_type_string(x), axis=1
    )
)

subject_level_df = subject_level_df.assign(
    model_version=subject_level_df.apply(lambda x: make_model_version_string(x), axis=1)
)
subject_level_ensemble_df = subject_level_ensemble_df.assign(
    model_version=subject_level_ensemble_df.apply(
        lambda x: make_model_version_string(x), axis=1
    )
)

# start plotting
metric_and_text = {
    "overall_precision": "Precision [0,1]",
    "overall_recall": "Recall [0,1]",
    "overall_accuracy": "Accuracy [0,1]",
    "overall_f1-score": "F1-score [0,1]",
    "overall_auc": "AUC [0,1]",
    "matthews_correlation_coefficient": "Matthews correlation coefficient [-1,1]",
}

metric_and_text = {
    "matthews_correlation_coefficient": "Matthews correlation coefficient [-1,1]"
}

hue_pretraining_type_order = ["ImageNet", "SimCLR_TCGA", "SimCLR_CBTN"]
# x_model_version_order = [
#     "ResNet50",
#     "ResNet50 (i+a) SAE",
#     "ResNet50 (i+a) LAE",
#     "ViT_b_16",
#     "ViT_b_16 (i+a) SAE",
#     "ViT_b_16 (i+a) LAE",
# ]

x_model_version_order = [
    "ResNet50",
    "ResNet50 (i+a) SAE",
    "ViT_b_16",
    "ViT_b_16 (i+a) SAE",
]
tick_font_size = 17
title_font_size = 20
label_font_size = 20
legend_font_size = 12

for metric_to_plot, metric_name_for_plot in metric_and_text.items():
    fig, box_plot = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

    box_plot = sns.boxplot(
        x="model_version",
        y=metric_to_plot,
        hue="pretraining_type_str",
        data=subject_level_df,
        palette="Set3",
        hue_order=hue_pretraining_type_order,
        order=x_model_version_order,
    )

    # ## fix title and axis
    box_plot.set_title(f"Classification performance", fontsize=title_font_size)
    box_plot.tick_params(labelsize=tick_font_size)
    box_plot.set_ylabel(
        f"{metric_name_for_plot}",
        fontsize=label_font_size,
    )
    # format y tick labels
    ylabels = [f"{x:,.2f}" for x in box_plot.get_yticks()]
    box_plot.set_yticklabels(ylabels)

    box_plot.set_xlabel("")
    box_plot.set_xticklabels(box_plot.get_xticklabels(), rotation=-15, ha="left")

    # ## add pattern to boxplots and legend
    available_hatches = [
        "\\",
        "xx",
        "/",
        "\\",
        "|",
        "-",
        ".",
        "oo",
        "\|",
        "/|",
        "-\ ",
        "-/",
    ]
    hatches = []
    hatches.extend(
        available_hatches[0 : len(hue_pretraining_type_order)]
        * len(x_model_version_order)
    )
    colors, legend_hatch = [], []

    for i, patch in enumerate(
        [patch for patch in box_plot.patches if type(patch) == mpl.patches.PathPatch]
    ):
        # Boxes from left to right
        hatch = hatches[i]
        patch.set_hatch(hatch)
        patch.set_edgecolor("k")
        patch.set_linewidth(1)
        colors.append(patch.get_facecolor())
        legend_hatch.append(hatch)

    # ## add ensemble
    add_ensemble_values(
        subject_level_ensemble_df,
        box_plot,
        x_model_version_order,
        hue_pretraining_type_order,
        metric_to_plot,
        numeric_value_settings={"fontsize": 12, "rotation": -45},
    )

    # ## fix legend
    legend_labels = [f"{l.get_text()}" for l in box_plot.legend_.texts]
    legend_colors = colors[:: len(colors) // len(hue_pretraining_type_order)]
    legend_hatch = legend_hatch[:: len(legend_hatch) // len(hue_pretraining_type_order)]

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
    if SAVE == True:
        file_name = f"Summary_classification_performance_{WHAT}_{metric_to_plot}"
        fig.savefig(
            os.path.join(SAVE_PATH, file_name + ".pdf"), bbox_inches="tight", dpi=100
        )
        fig.savefig(
            os.path.join(SAVE_PATH, file_name + ".png"), bbox_inches="tight", dpi=100
        )
        plt.close(fig)
    else:
        plt.show()

# %% ON THE SAME MODEL VERSION AND PRE-TRAINING, COMPARE THE PERFORMANCE WHEN USING OR NOT THE SUBJECT AGE.
# THIS IS A SUBJECT WISE COMPARISON BETWEEN TWO MODELS ON THE SAME DATA. HERE WE WILL USE THE NON-PARAMETRIC
# WILCOXON SIGNED-RAKED TEST BETWEEN THE THE POPULATION OF SUBJECT-WISE PERFORMANCE FOR EACH FOLD (50 IN TOTAL
# GIVEN 10 TIMES REPEATED 5 FOLD CROSS VALIDATION) WHEN NOT USING SUBJECT AGE AND WHEN USING SUBJECT AGE.

# Here for each model type and for each pretraining configuration, select the teo populations (without and with age metrics)
# perform the comparison and print the results.
unique_things_to_compare = pd.unique(subject_level_df.use_age)
list_of_comparisons = list(itertools.combinations(unique_things_to_compare, 2))
significance_thr = 0.05
bonferroni_corrected_significance_thr = significance_thr / len(list_of_comparisons)

# summary for saving
significant_test_analysis_summary = []
significant_test_analysis_summary.append(
    f"Subject-wise statistical analysis of age-vs-noAge for {MR_SEQUENCE} MR sequence."
)
significant_test_analysis_summary.append(
    f"Using a significance threshold of {significance_thr:0.4f} adjusted using bonferoni correction for multiple comparisons ({len(list_of_comparisons)} comparisons -> {bonferroni_corrected_significance_thr:0.4f} corrected threshold)."
)


# select mr sequence
DF = ORIGINAL_DF.loc[ORIGINAL_DF["mr_sequence"] == MR_SEQUENCE]
# take subject-wise prediction
FILTER = [f"subject_ensemble_pred_fold_{i+1}" for i in range(20)]
subject_level_df = DF[DF["performance_over"].isin(FILTER)]
# make an easy to use string for the pretraining type
subject_level_df = subject_level_df.assign(
    pretraining_type_str=subject_level_df.apply(
        lambda x: make_pretraining_type_string(x), axis=1
    )
)

metrics_to_test = {
    "overall_accuracy": "Accuracy",
    "overall_auc": "AUC",
    "matthews_correlation_coefficient": "Matthews correlation coefficient",
}

metrics_to_test = {
    "matthews_correlation_coefficient": "Matthews correlation coefficient",
}

string = f"MR sequence: {MR_SEQUENCE:3s}"
significant_test_analysis_summary.append(string)
for model_version in pd.unique(subject_level_df.model_version):
    string = f"{' '*2:s}Model version: {model_version:{np.max([len(v) for v in pd.unique(subject_level_df.model_version)])}s}"
    significant_test_analysis_summary.append(string)
    for pre_training_version in pd.unique(subject_level_df.pretraining_type_str):
        string = f"{' '*4:s}Pretraining version: {pre_training_version:{np.max([len(v) for v in pd.unique(subject_level_df.pretraining_type_str)])}s}"
        significant_test_analysis_summary.append(string)
        for metric, metric_full_name in metrics_to_test.items():
            string = f"{' '*6:s} Metric: {metric:{np.max([len(v) for v in metrics_to_test.keys()])}s}"
            significant_test_analysis_summary.append(string)
            # get the two populations
            population_1 = list(
                subject_level_df.loc[
                    (subject_level_df.use_age == False)
                    & (subject_level_df.model_version == model_version)
                    & (subject_level_df.pretraining_type_str == pre_training_version)
                ][metric]
            )
            population_2 = list(
                subject_level_df.loc[
                    (subject_level_df.use_age == True)
                    & (subject_level_df.model_version == model_version)
                    & (subject_level_df.pretraining_type_str == pre_training_version)
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
                string = f"{' '*8:s}Population 1 (age==False) [mean±std]: {len(population_1)} samples, {np.mean(population_1):0.4f} ± {np.std(population_1):0.4f}"
                significant_test_analysis_summary.append(string)
                string = f"{' '*8:s}Population 2 (age==True) [mean±std]: {len(population_2)} samples, {np.mean(population_2):0.4f} ± {np.std(population_2):0.4f}"
                significant_test_analysis_summary.append(string)
                string = f"{' '*8:s}p-value: {statistical_test[-1]:0.8f} ({'SIGNIFICANT' if statistical_test[-1] <= bonferroni_corrected_significance_thr else 'NOT SIGNIFICANT'})"
                significant_test_analysis_summary.append(string)
            else:
                string = f"{' '*8:s}Skipping statistical test since len populaiton_1=={len(population_1)} and len populaiton_2=={len(population_2)}"
                significant_test_analysis_summary.append(string)
    significant_test_analysis_summary.append("\n")
# print summary
for s in significant_test_analysis_summary:
    print(s)
# save to file
summary_file = os.path.join(
    SAVE_PATH, f"statistica_analysis_summary_{MR_SEQUENCE}_age_vs_noAge.txt"
)
with open(summary_file, "w") as f:
    for line in significant_test_analysis_summary:
        f.write(f"{line}\n")

# %% ON THE SAME MODEL VERSION AND INPUT TYPE (WITH AND WITHOUT), TEST IF MODEL TRETRAINING DID MAKE ANY DIFFERENCE.
# THIS IS A SUBJECT WISE COMPARISON BETWEEN TWO MODELS ON THE SAME DATA. HERE WE USE THE NON-PARAMETRIC
# WILCOXON SIGNED-RAKED TEST BETWEEN THE THE POPULATION OF SUBJECT-WISE PERFORMANCE FOR EACH FOLD (50 IN TOTAL
# GIVEN 10 TIMES REPEATED 5 FOLD CROSS VALIDATION) WHEN THE MODELS ARE PRETRAINED ON IMAGENET, SIMCLR_TCGA AND SIMCLR_CBTN.
unique_things_to_compare = pd.unique(subject_level_df.pretraining_type_str)
list_of_comparisons = list(itertools.combinations(unique_things_to_compare, 2))
significance_thr = 0.05
bonferroni_corrected_significance_thr = significance_thr / len(list_of_comparisons)

# summary for saving
significant_test_analysis_summary = []
significant_test_analysis_summary.append(
    f"Subject-wise statistical analysis between pretraining versions ({unique_things_to_compare}) for {MR_SEQUENCE} MR sequence."
)
significant_test_analysis_summary.append(
    f"Using a significance threshold of {significance_thr:0.4f} adjusted using bonferoni correction for multiple comparisons ({len(list_of_comparisons)} comparisons -> {bonferroni_corrected_significance_thr:0.4f} corrected threshold)."
)

# select mr sequence
DF = ORIGINAL_DF.loc[ORIGINAL_DF["mr_sequence"] == MR_SEQUENCE]
# take subject-wise prediction
FILTER = [f"subject_ensemble_pred_fold_{i+1}" for i in range(20)]
subject_level_df = DF[DF["performance_over"].isin(FILTER)]
# make an easy to use string for the pretraining type
subject_level_df = subject_level_df.assign(
    pretraining_type_str=subject_level_df.apply(
        lambda x: make_pretraining_type_string(x), axis=1
    )
)

metrics_to_test = {
    "overall_accuracy": "Accuracy",
    "overall_auc": "AUC",
    "matthews_correlation_coefficient": "Matthews correlation coefficient",
}

metrics_to_test = {
    "matthews_correlation_coefficient": "Matthews correlation coefficient",
}

string = f"MR sequence: {MR_SEQUENCE:3s}"
significant_test_analysis_summary.append(string)
for model_version in pd.unique(subject_level_df.model_version):
    string = f"{' '*2:s}Model version: {model_version:{np.max([len(v) for v in pd.unique(subject_level_df.model_version)])}s}"
    significant_test_analysis_summary.append(string)
    for use_age in [True, False]:
        string = f"{' '*4:s}Using age: {use_age}"
        significant_test_analysis_summary.append(string)
        for metric, metric_full_name in metrics_to_test.items():
            string = f"{' '*6:s} Metric: {metric:{np.max([len(v) for v in metrics_to_test.keys()])}s}"
            significant_test_analysis_summary.append(string)
            for things_to_compare in list_of_comparisons:
                string = f"{' '*8:s} Comparing: {things_to_compare[0]:{np.max([len(v) for v in unique_things_to_compare])}s} - vs - {things_to_compare[1]:{np.max([len(v) for v in unique_things_to_compare])}s}"
                significant_test_analysis_summary.append(string)
                # get the two populations
                population_1 = list(
                    subject_level_df.loc[
                        (subject_level_df.use_age == use_age)
                        & (subject_level_df.model_version == model_version)
                        & (
                            subject_level_df.pretraining_type_str
                            == things_to_compare[0]
                        )
                    ][metric]
                )

                population_2 = list(
                    subject_level_df.loc[
                        (subject_level_df.use_age == use_age)
                        & (subject_level_df.model_version == model_version)
                        & (
                            subject_level_df.pretraining_type_str
                            == things_to_compare[1]
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
                    indent = 10
                    string = f"{' '*indent:s}Population 1 ({things_to_compare[0]}) [mean±std]: {len(population_1)} samples, {np.mean(population_1):0.4f} ± {np.std(population_1):0.4f}"
                    significant_test_analysis_summary.append(string)
                    string = f"{' '*indent:s}Population 2 ({things_to_compare[1]}) [mean±std]: {len(population_2)} samples, {np.mean(population_2):0.4f} ± {np.std(population_2):0.4f}"
                    significant_test_analysis_summary.append(string)
                    string = f"{' '*indent:s}p-value: {statistical_test[-1]:0.8f} ({'SIGNIFICANT' if statistical_test[-1] <= bonferroni_corrected_significance_thr else 'NOT SIGNIFICANT'})"
                    significant_test_analysis_summary.append(string)
                else:
                    string = f"{' '*indent:s}Skipping statistical test since len populaiton_1=={len(population_1)} and len populaiton_2=={len(population_2)}"
                    significant_test_analysis_summary.append(string)
    significant_test_analysis_summary.append("\n")

# print summary
for s in significant_test_analysis_summary:
    print(s)
# save to file
summary_file = os.path.join(
    SAVE_PATH, f"statistica_analysis_summary_{MR_SEQUENCE}_PreTraining_version.txt"
)
with open(summary_file, "w") as f:
    for line in significant_test_analysis_summary:
        f.write(f"{line}\n")

# %% USING THE BEST MODEL FOR EACH MODEL VERSION (AMONG THE PRE-TRAINING VERSION AND USE OR NOT AGE),
# PERFORM A SUBJECT-WISE COMPARISON BETWEEN THEM.
# THIS IS A SUBJECT WISE COMPARISON BETWEEN TWO MODELS' PERFORMANCES. HERE WE USE THE NON-PARAMETRIC
# WILCOXON SIGNED-RAKED TEST BETWEEN THE THE POPULATION OF SUBJECT-WISE PERFORMANCE FOR EACH FOLD (50 IN TOTAL
# GIVEN 10 TIMES REPEATED 5 FOLD CROSS VALIDATION).

# select mr sequence
DF = ORIGINAL_DF.loc[ORIGINAL_DF["mr_sequence"] == MR_SEQUENCE]
# take subject-wise prediction
FILTER = [f"subject_ensemble_pred_fold_{i+1}" for i in range(20)]
subject_level_df = DF[DF["performance_over"].isin(FILTER)]
# make an easy to use string for the pretraining type
subject_level_df = subject_level_df.assign(
    pretraining_type_str=subject_level_df.apply(
        lambda x: make_pretraining_type_string(x), axis=1
    )
)

# metrics to test
metrics_to_test = {
    "overall_accuracy": "Accuracy",
    "overall_auc": "AUC",
    "matthews_correlation_coefficient": "Matthews correlation coefficient",
}

# metrics_to_test = {
#     "matthews_correlation_coefficient": "Matthews correlation coefficient",
# }

# comparison settings
unique_things_to_compare = pd.unique(subject_level_df.model_version)
list_of_comparisons = list(itertools.combinations(unique_things_to_compare, 2))
significance_thr = 0.05
bonferroni_corrected_significance_thr = significance_thr / len(list_of_comparisons)

# summary for saving
significant_test_analysis_summary = []
significant_test_analysis_summary.append(
    f"Subject-wise statistical analysis between the best performing models for each model version ({unique_things_to_compare}) for {MR_SEQUENCE} MR sequence."
)
significant_test_analysis_summary.append(
    f"Using a significance threshold of {significance_thr:0.4f} adjusted using bonferoni correction for multiple comparisons ({len(list_of_comparisons)} comparisons -> {bonferroni_corrected_significance_thr:0.4f} corrected threshold)."
)

# find the best performing model pretraining and input for every model version
best_settings = dict.fromkeys(unique_things_to_compare)

for metric, metric_full_name in metrics_to_test.items():
    string = f"Metric: {metric_full_name}"
    significant_test_analysis_summary.append(string)
    for model_version in unique_things_to_compare:
        best_settings[model_version] = {
            "pre_training": [],
            "use_age": [],
            "mean": 0,
            "std": 0,
        }
        for pre_training_version in pd.unique(subject_level_df.pretraining_type_str):
            for use_age in [True, False]:
                # get mean (and std) for this metric and this model configuration
                mean = np.mean(
                    subject_level_df.loc[
                        (subject_level_df.use_age == use_age)
                        & (subject_level_df.model_version == model_version)
                        & (
                            subject_level_df.pretraining_type_str
                            == pre_training_version
                        )
                    ][metric]
                )
                std = np.std(
                    subject_level_df.loc[
                        (subject_level_df.use_age == use_age)
                        & (subject_level_df.model_version == model_version)
                        & (
                            subject_level_df.pretraining_type_str
                            == pre_training_version
                        )
                    ][metric]
                )
                # save info (just to check that the best was chosen)
                string = f"{model_version:{np.max([len(v) for v in unique_things_to_compare])}s}, {pre_training_version:{np.max([len(v) for v in pd.unique(subject_level_df.pretraining_type_str)])}s}, use_Age {str(use_age):5s}: {mean:0.4f} ± {std:0.4f}"
                significant_test_analysis_summary.append(string)

                # check if better than the currect best
                if mean > best_settings[model_version]["mean"]:
                    # update best values
                    best_settings[model_version]["pre_training"] = pre_training_version
                    best_settings[model_version]["use_age"] = use_age
                    best_settings[model_version]["mean"] = mean
                    best_settings[model_version]["std"] = std
    # just for formatting
    significant_test_analysis_summary.append("\n")
    # get the populations from the best performing models for all the versions
    populations = []
    for model_version, settings in best_settings.items():
        string = f'Best model for {model_version} is when using: {settings["pre_training"]}, use_age {settings["use_age"]} ({settings["mean"]:0.4f} ± {settings["std"]:0.4f})'
        significant_test_analysis_summary.append(string)
        # get values fr these settings
        populations.append(
            list(
                subject_level_df.loc[
                    (subject_level_df.use_age == settings["use_age"])
                    & (subject_level_df.model_version == model_version)
                    & (
                        subject_level_df.pretraining_type_str
                        == settings["pre_training"]
                    )
                ][metric]
            )
        )
    # perform comparison
    population_1 = populations[0]
    population_2 = populations[1]
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
        indent = 0
        string = f"{' '*indent:s}Population 1 ({list(best_settings.keys())[0]}) [mean±std]: {len(population_1)} samples, {np.mean(population_1):0.4f} ± {np.std(population_1):0.4f}"
        significant_test_analysis_summary.append(string)
        string = f"{' '*indent:s}Population 2 ({list(best_settings.keys())[1]}) [mean±std]: {len(population_2)} samples, {np.mean(population_2):0.4f} ± {np.std(population_2):0.4f}"
        significant_test_analysis_summary.append(string)
        string = f"{' '*indent:s}p-value: {statistical_test[-1]:0.8f} ({'SIGNIFICANT' if statistical_test[-1] <= bonferroni_corrected_significance_thr else 'NOT SIGNIFICANT'})"
        significant_test_analysis_summary.append(string)
    else:
        string = f"{' '*indent:s}Skipping statistical test since len populaiton_1=={len(population_1)} and len populaiton_2=={len(population_2)}"
        significant_test_analysis_summary.append(string)

    # just for formatting
    significant_test_analysis_summary.append("\n")

# print summary
for s in significant_test_analysis_summary:
    print(s)
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

# %% COMPUTE THE CLASSIFICATION PERORMANCE WHEN USING THE PREDICTIONS FROM ALL THE AVAILABLE MR MODALITIES
# HERE WE WORK ON A SUBJECT-LEVEL PREDICTION SINCE THE SLICES OF EACH MODALITY ARE NOT ALLIGNED.
# EXPLORE HOW THE ADDITION OF MODEL-FOLDS AND MODEL-MODALITY CHANGES THE OVERALL PREDICTION.
# WE CAN USE THE PLOT FUNCTIONS AS FROM BEFORE WITH THE ADDED TYPES OF MODELS (EG. RESNET50_T1_T2_ADC with the values for only the subject level prediction)

# get the list of unique subject IDs for all the unique modalities
unique_subjects_per_modality = [
    list(
        pd.unique(
            DETAILED_DF_ORIGINAL.loc[DETAILED_DF_ORIGINAL.mr_sequence == mr_sequence][
                "subject_IDs"
            ]
        )
    )
    for mr_sequence in pd.unique(DETAILED_DF_ORIGINAL.mr_sequence)
]
unique_overlapping_subjects = set.intersection(
    *[set(list_) for list_ in unique_subjects_per_modality]
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
MODEL_VERSIONS = ["ResNet50"] * len(MR_SEQUENCES_COMBINATIONS)
PRE_TRAINING_VERSIONS = ["ImageNet"] * len(MR_SEQUENCES_COMBINATIONS)
USE_AGE = [False] * len(MR_SEQUENCES_COMBINATIONS)

# start the looping
for indx_n, n in enumerate(range(NUMBER_ORANDOM_SAMPLINGS)):
    for mr_sequence_combination in MR_SEQUENCES_COMBINATIONS:
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
        # initiate list that hold the infromation for plotting later
        aus_df = []

        # for each combination of mr sequence, model version and model selection critaria, get indexes from the pool of models
        for mr_seq, model_version, pre_training, use_age in zip(
            mr_sequence_combination, MODEL_VERSIONS, PRE_TRAINING_VERSIONS, USE_AGE
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
                        values = values.groupby("subject_IDs").mean()
                        # get metrics and save


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
        x[
            f"subject_entropy_over_the_folds_class_{c+1}"
        ] = x.subject_entropy_over_the_folds[c]

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
