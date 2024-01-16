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
    boxes = [c for c in ax.get_children() if type(c).__name__ == "PathPatch"]
    lines_per_box = int(len(lines) / len(boxes))

    # define markers to use
    available_markers = ["s", "v", "^", "p", "X", "8", "*"]
    # index cycle. This is a litle bity tricky since the boxplot orted follows the hue order and then the x order.
    index_hue_cycle = []
    [index_hue_cycle.extend([i] * len(x_order)) for i in range(len(hue_order))]
    index_x_cycle = [i for i in range(len(x_order))] * len(hue_order)
    # [index_x_cycle.extend([i] * len(hue_order)) for i in range(len(x_order))]

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
            # add value cose to the parcker in a box
            text_box = dict(boxstyle="round", facecolor="white", alpha=0.7)
            if numeric_value_settings:
                ax.text(
                    x=x,
                    y=y - 0.05,
                    s=f"{y:0.2f}",
                    zorder=6,
                    bbox=text_box,
                    **numeric_value_settings,
                )
            else:
                # use default
                ax.text(
                    x=x,
                    y=y - 0.05,
                    s=f"{y:0.2f}",
                    zorder=6,
                    c="k",
                    fontsize=25,
                    bbox=text_box,
                    path_effects=[pe.withStroke(linewidth=2, foreground="ghostwhite")],
                )


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
SUMMARY_FILE_PATH = "/Users/iulta54/Desktop/CBTN_v1/for_project_summary_20231208/evaluation_aggregated.xlsx"
SAVE_PATH = pathlib.Path(
    os.path.join(os.path.dirname(SUMMARY_FILE_PATH), "Summary_plots")
)
SAVE_PATH.mkdir(parents=True, exist_ok=True)
DF = pd.read_excel(SUMMARY_FILE_PATH)

# use model with 0.5 fine tuning
DF = DF.loc[DF.fine_tuning == 0.5]

# %% PLOT SLICE-SIZE BOXPLOTS FOR THE DIFFERENT MDOEL VERSIONS USING AS GROUPING THE DIFFERENT PRE-TRAINING TYPES
# Filter the dataframe to only get the metrics for the different folds of the repetitions on a slide level

SAVE = False
USE_AGE = False
WHAT = "slice_wise_no_age"

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
metrics = [
    "overall_precision",
    "overall_recall",
    "overall_accuracy",
    "overall_f1-score",
    "overall_auc",
    "matthews_correlation_coefficient",
]
# metrics = ['matthews_correlation_coefficient']
metric_name_for_plot = [
    "Precision [0,1]",
    "Recall [0,1]",
    "Accuracy [0,1]",
    "F1-score [0,1]",
    "AUC [0,1]",
    "Matthews correlation coefficient [-1,1]",
]

hue_pretraining_type_order = ["ImageNet", "SimCLR_TCGA", "SimCLR_CBTN"]
x_model_version_order = ["ResNet50", "ViT_b_16"]

tick_font_size = 17
title_font_size = 20
label_font_size = 20
legend_font_size = 12

for metric_to_plot in metrics:
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
        f"{metric_name_for_plot[metrics.index(metric_to_plot)]}",
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

# %% PLOT SUBJECT-WISE BOXPLOTS FOR THE DIFFERENT MDOEL VERSIONS USING AS GROUPING THE DIFFERENT PRE-TRAINING TYPES
# Filter the dataframe to only get the metrics for the different folds of the repetitions on a slide level

SAVE = True
USE_AGE = False
WHAT = "subject_wise_no_age"

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
metrics = [
    "precision",
    "recall",
    "accuracy",
    "f1-score",
    "auc",
    "matthews_correlation_coefficient",
]
metrics = [
    "overall_precision",
    "overall_recall",
    "overall_accuracy",
    "overall_f1-score",
    "overall_auc",
    "matthews_correlation_coefficient",
]
hue_pretraining_type_order = ["ImageNet", "SimCLR_TCGA", "SimCLR_CBTN"]
x_model_version_order = ["ResNet50", "ViT_b_16"]

tick_font_size = 17
title_font_size = 20
label_font_size = 20
legend_font_size = 12

for metric_to_plot in metrics:
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
        f"{metric_name_for_plot[metrics.index(metric_to_plot)]}",
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

SAVE = True
WHAT = "slice_wise"

FILTER = [f"pred_fold_{i+1}" for i in range(20)]
slice_level_df = DF[DF["performance_over"].isin(FILTER)]
slice_level_ensemble_df = DF.loc[(DF.performance_over == "per_slice_ensemble")]

# Create s string for the different types of pretraining
# pretraining==False -> ImageNet
# pretraining==True $ pretraining_dataset == tcga -> SimCLR_TCGA
# pretraining==True $ pretraining_dataset == cbtn -> SimCLR_CBTN


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


slice_level_df["pretraining_type_str"] = slice_level_df.apply(
    lambda x: make_pretraining_type_string(x), axis=1
)
slice_level_ensemble_df["pretraining_type_str"] = slice_level_ensemble_df.apply(
    lambda x: make_pretraining_type_string(x), axis=1
)

slice_level_df["model_version"] = slice_level_df.apply(
    lambda x: make_model_version_string(x), axis=1
)
slice_level_ensemble_df["model_version"] = slice_level_ensemble_df.apply(
    lambda x: make_model_version_string(x), axis=1
)


# start plotting
metrics = [
    "overall_precision",
    "overall_recall",
    "overall_accuracy",
    "overall_f1-score",
    "overall_auc",
    "matthews_correlation_coefficient",
]
metric_name_for_plot = [
    "Precision [0,1]",
    "Recall [0,1]",
    "Accuracy [0,1]",
    "F1-score [0,1]",
    "AUC [0,1]",
    "Matthews correlation coefficient [-1,1]",
]

hue_pretraining_type_order = ["ImageNet", "SimCLR_TCGA", "SimCLR_CBTN"]
x_model_version_order = [
    "ResNet50",
    "ResNet50 (i+a) SAE",
    "ResNet50 (i+a) LAE",
    "ViT_b_16",
    "ViT_b_16 (i+a) SAE",
    "ViT_b_16 (i+a) LAE",
]
tick_font_size = 17
title_font_size = 20
label_font_size = 20
legend_font_size = 12

for metric_to_plot in metrics:
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
        f"{metric_name_for_plot[metrics.index(metric_to_plot)]}",
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

SAVE = True
WHAT = "subject_wise"

FILTER = [f"subject_ensemble_pred_fold_{i+1}" for i in range(20)]
subject_level_df = DF[DF["performance_over"].isin(FILTER)]
subject_level_ensemble_df = DF.loc[(DF.performance_over == "overall_subject_ensemble")]

# Create s string for the different types of pretraining
# pretraining==False -> ImageNet
# pretraining==True $ pretraining_dataset == tcga -> SimCLR_TCGA
# pretraining==True $ pretraining_dataset == cbtn -> SimCLR_CBTN


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


subject_level_df["pretraining_type_str"] = subject_level_df.apply(
    lambda x: make_pretraining_type_string(x), axis=1
)
subject_level_ensemble_df["pretraining_type_str"] = subject_level_ensemble_df.apply(
    lambda x: make_pretraining_type_string(x), axis=1
)

subject_level_df["model_version"] = subject_level_df.apply(
    lambda x: make_model_version_string(x), axis=1
)
subject_level_ensemble_df["model_version"] = subject_level_ensemble_df.apply(
    lambda x: make_model_version_string(x), axis=1
)


# start plotting
metrics = [
    "overall_precision",
    "overall_recall",
    "overall_accuracy",
    "overall_f1-score",
    "overall_auc",
    "matthews_correlation_coefficient",
]
metric_name_for_plot = [
    "Precision [0,1]",
    "Recall [0,1]",
    "Accuracy [0,1]",
    "F1-score [0,1]",
    "AUC [0,1]",
    "Matthews correlation coefficient [-1,1]",
]

hue_pretraining_type_order = ["ImageNet", "SimCLR_TCGA", "SimCLR_CBTN"]
x_model_version_order = [
    "ResNet50",
    "ResNet50 (i+a) SAE",
    "ResNet50 (i+a) LAE",
    "ViT_b_16",
    "ViT_b_16 (i+a) SAE",
    "ViT_b_16 (i+a) LAE",
]
tick_font_size = 17
title_font_size = 20
label_font_size = 20
legend_font_size = 12

for metric_to_plot in metrics:
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
        f"{metric_name_for_plot[metrics.index(metric_to_plot)]}",
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


# %% PLOT DETAILED EVALUATION
SUMMARY_FILE_PATH = "/Users/iulta54/Desktop/CBTN_v1/for_project_summary_20231208/detailed_evaluation_aggregated.csv"
SAVE_PATH = pathlib.Path(
    os.path.join(os.path.dirname(SUMMARY_FILE_PATH), "Summary_plots")
)
SAVE_PATH.mkdir(parents=True, exist_ok=True)
DF = pd.read_csv(SUMMARY_FILE_PATH, low_memory=False)

# use model with 0.5 fine tuning
DF = DF.loc[DF.fine_tuning == 0.5]

# %% PLOT THE PER MODEL (with and withouth age), PER-FOLD AND PER CLASS ENTROPY - THIS IS A PER-SUBJECT EVALUATION SINCE WE ARE
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
