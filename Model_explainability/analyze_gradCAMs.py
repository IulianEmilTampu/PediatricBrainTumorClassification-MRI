#%%
"""
This script reads the files computed using the get_GradCAMs_from_cv_models.py
and plots the gradCAM map and performs computation on them.

The script works on .npy files created using the aforementioned script. Each file contains
a dictionary structured as follows:

- anatomical_image : np.array
    This is the anatomical image (the first channel if multi-channel) used for plotting
- anatomical_image_modality : str
    Which modality is the anatomical image.
- original_image_prediction_distribution : np.array
    Array of size [N, C] containing the logits of every model (N) for all the classes (C)
- gradCAM_raw : np.array
    Array of size [N,W,H,Ch,L] containing the raw gradCAM for an image at every location (W, H)
    for evey class (Ch), every model (N) and for every convolutional layer (L)
- gradCAM_rgb : np.array
    Array of size [N,W,H,3(rgb),Ch,L] containing the rbg gradCAM for an image at every location (W, H)
    for evey class (Ch), every model (N) and for every convolutional layer (L)
- layer_names : list
    List of strings containing the names of the convolutional layers
"""

import os
import glob
import pathlib
import argparse
import numpy as np
import pandas as pd
from scipy import ndimage
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.gridspec as gridspec

#%% DEFINE UTILITIES

# plotting utilities
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    from mpl_toolkits import axes_grid1

    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def plot_gradCAM(
    gradCAM_dictionary,
    class_of_interest=1,
    layer_of_interest=-1,
    attribution_thr=0.8,
    scale=True,
    plot_annotation_countour=True,
    save_path=None,
    draw=False,
    visual_settings_dict=None,
):
    """
    Utility that plots and saves, if requested, the GradCAMs.
    """

    foreground_class = class_of_interest
    background_class = 0 if class_of_interest == 1 else 1

    plt.rcParams["font.family"] = "Times New Roman"

    default_visual_settings_dict = {
        "title_font_size": 20,
        "colorbar_font": 15,
        "cmap_mean_gradCAM": "jet",
        "cmap_STD_gradCAM": "jet",
        "cmap_annotation_contour": "hsv",
        "cmap_brain_contour": "gray",
        "space_between_rows": 0.05,
        "start_upper_row": 0.7,
        "dpi": 100,
    }
    # build visual_settings_dict if not given
    if not visual_settings_dict:
        visual_settings_dict = default_visual_settings_dict
    else:
        # check which parametrs are given and fix default for those not given
        for key, value in default_visual_settings_dict.items():
            if key not in list(visual_settings_dict.keys()):
                visual_settings_dict[key] = value

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(40, 25), facecolor="w")

    # ######################### plot anatomical image
    # fix anatomical image due to transpose in the occlusion value saving
    anatomical_img = np.fliplr(np.rot90(gradCAM_dictionary["anatomical_image"], k=3))
    aus_ax = ax[0, 0]
    im = aus_ax.imshow(
        anatomical_img,
        cmap="gray",
        interpolation=None,
    )
    aus_ax.axis("off")
    add_colorbar(im)
    mean_logits = np.mean(
        gradCAM_dictionary["original_image_prediction_distribution"], axis=0
    )
    # print GT based on of it is a normal image or a FOCUS metric image
    if "focus_metrics" in list(gradCAM_dictionary.keys()):
        # print gt
        if gradCAM_dictionary["focus_metrics"]:
            gt = gradCAM_dictionary["ground_truth"]
            aus_ax.set_title(
                f"Anatomical image (GT={gt})\n (TB_LR)",
                fontsize=visual_settings_dict["title_font_size"],
            )
    else:
        gt = np.argmax(gradCAM_dictionary["ground_truth"])
        aus_ax.set_title(
            f"Anatomical image (GT={gt:0.1f})\nUnpathced logits{mean_logits}",
            fontsize=visual_settings_dict["title_font_size"],
        )

    # plot the annotation countour if gt is 1 (slice with tumor)
    if any(
        [
            plot_annotation_countour,
            gt == 1,
            "focus_metrics" in list(gradCAM_dictionary.keys()),
        ]
    ):
        annotation = (np.copy(gradCAM_dictionary["annotation_image"]) > 0.5).astype(int)
        annotation = np.fliplr(np.rot90(annotation, k=3))
        countour = annotation - ndimage.binary_erosion(annotation, iterations=4)
        countour = np.ma.masked_where(countour == 0, countour)

    # ######################### plot anatomical image with annotation contour
    aus_ax = ax[1, 0]
    im = aus_ax.imshow(
        anatomical_img,
        cmap="gray",
        interpolation=None,
    )
    aus_ax.axis("off")
    add_colorbar(im)
    mean_logits = np.mean(
        gradCAM_dictionary["original_image_prediction_distribution"], axis=0
    )

    # plot the annotation countour if gt is 1 (slice with tumor)
    if any(
        [
            plot_annotation_countour,
            gt == 1,
            "focus_metrics" in list(gradCAM_dictionary.keys()),
        ]
    ):
        aus_ax.imshow(countour, cmap=visual_settings_dict["cmap_annotation_contour"])

    # ######################### mean GradCAM for the foregroung (tumor) class
    to_plot = gradCAM_dictionary["gradCAM_raw"][
        :, :, :, foreground_class, layer_of_interest
    ].squeeze()

    if scale:
        vmin = 0
        vmax = 1
        # normalize grad cam
        to_plot = (to_plot - to_plot.min()) / (to_plot.max() - to_plot.min())
    else:
        vmin = to_plot.min()
        vmax = to_plot.max()

    if len(to_plot.shape) != 2:
        to_plot = np.fliplr(np.rot90(to_plot.mean(axis=0), k=3))
    else:
        to_plot = np.fliplr(np.rot90(to_plot, k=3))

    # save for later
    gradCam_mean_map = to_plot

    aus_ax = ax[0, 1]
    im = aus_ax.imshow(
        to_plot,
        cmap=visual_settings_dict["cmap_mean_gradCAM"],
        interpolation=None,
        vmin=vmin,
        vmax=vmax,
    )
    aus_ax.axis("off")
    aus_ax.set_title(
        f"Mean GradCAM class of interest (# {foreground_class}, layer nbr. {layer_of_interest})",
        fontsize=visual_settings_dict["title_font_size"],
    )
    # add colorbar
    cbar = add_colorbar(im, aspect=20, pad_fraction=0.5)
    ticks = list(np.linspace(vmin, vmax, 6))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{i:0.2f}" for i in ticks])
    cbar.ax.tick_params(labelsize=visual_settings_dict["colorbar_font"])

    # ######################### mean GradCAM for the foregroung (tumor) class + annotation
    aus_ax = ax[1, 1]
    im = aus_ax.imshow(
        gradCam_mean_map,
        cmap=visual_settings_dict["cmap_mean_gradCAM"],
        interpolation=None,
        vmin=vmin,
        vmax=vmax,
    )
    aus_ax.axis("off")
    aus_ax.set_title(
        f"Mean GradCAM class of interest (# {foreground_class}, layer nbr. {layer_of_interest})",
        fontsize=visual_settings_dict["title_font_size"],
    )
    # add colorbar
    cbar = add_colorbar(im, aspect=20, pad_fraction=0.5)
    ticks = list(np.linspace(vmin, vmax, 6))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{i:0.2f}" for i in ticks])
    cbar.ax.tick_params(labelsize=visual_settings_dict["colorbar_font"])

    # plot annotation
    if any(
        [
            plot_annotation_countour,
            gt == 1,
            "focus_metrics" in list(gradCAM_dictionary.keys()),
        ]
    ):
        aus_ax.imshow(countour, cmap=visual_settings_dict["cmap_annotation_contour"])

    # ######################### STD gradCAM foreground (class of interest)
    to_plot = gradCAM_dictionary["gradCAM_raw"][
        :, :, :, foreground_class, layer_of_interest
    ].squeeze()

    if scale:
        vmin = 0
        vmax = 1
        # normalize grad cam
        to_plot = (to_plot - to_plot.min()) / (to_plot.max() - to_plot.min())
    else:
        vmin = to_plot.min()
        vmax = to_plot.max()

    if len(to_plot.shape) != 2:
        to_plot = np.fliplr(np.rot90(to_plot.std(axis=0), k=3))
    else:
        to_plot = np.fliplr(np.rot90(to_plot, k=3))

    # save for later
    gradCam_STD_map = to_plot

    aus_ax = ax[0, 2]
    im = aus_ax.imshow(
        to_plot,
        cmap=visual_settings_dict["cmap_STD_gradCAM"],
        interpolation=None,
        vmin=vmin,
        vmax=vmax,
    )
    aus_ax.axis("off")
    aus_ax.set_title(
        f"STD GradCAM for class of interest (# {class_of_interest})",
        fontsize=visual_settings_dict["title_font_size"],
    )
    # add colorbar
    cbar = add_colorbar(im, aspect=20, pad_fraction=0.5)
    ticks = list(np.linspace(vmin, vmax, 6))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{i:0.2f}" for i in ticks])
    cbar.ax.tick_params(labelsize=visual_settings_dict["colorbar_font"])

    # ######################### STD gradCAM foreground (class of interest) + annotation
    aus_ax = ax[1, 2]
    im = aus_ax.imshow(
        gradCam_STD_map,
        cmap=visual_settings_dict["cmap_STD_gradCAM"],
        interpolation=None,
        vmin=vmin,
        vmax=vmax,
    )
    aus_ax.axis("off")
    aus_ax.set_title(
        f"STD GradCAM for class of interest (# {class_of_interest})",
        fontsize=visual_settings_dict["title_font_size"],
    )
    # add colorbar
    cbar = add_colorbar(im, aspect=20, pad_fraction=0.5)
    ticks = list(np.linspace(vmin, vmax, 6))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{i:0.2f}" for i in ticks])
    cbar.ax.tick_params(labelsize=visual_settings_dict["colorbar_font"])

    if any(
        [
            plot_annotation_countour,
            gt == 1,
            "focus_metrics" in list(gradCAM_dictionary.keys()),
        ]
    ):
        aus_ax.imshow(countour, cmap=visual_settings_dict["cmap_annotation_contour"])

    # ######################### mean gradCAM background
    to_plot = gradCAM_dictionary["gradCAM_raw"][
        :, :, :, background_class, layer_of_interest
    ].squeeze()

    if scale:
        vmin = 0
        vmax = 1
        # normalize grad cam
        to_plot = (to_plot - to_plot.min()) / (to_plot.max() - to_plot.min())
    else:
        vmin = to_plot.min()
        vmax = to_plot.max()

    if len(to_plot.shape) != 2:
        to_plot = np.fliplr(np.rot90(to_plot.mean(axis=0), k=3))
    else:
        to_plot = np.fliplr(np.rot90(to_plot, k=3))

    aus_ax = ax[0, 3]
    im = aus_ax.imshow(
        to_plot,
        cmap=visual_settings_dict["cmap_mean_gradCAM"],
        interpolation=None,
        vmin=vmin,
        vmax=vmax,
    )
    aus_ax.axis("off")
    aus_ax.set_title(
        f"Mean GradCAM of background (# {background_class}, layer nbr. {layer_of_interest})",
        fontsize=visual_settings_dict["title_font_size"],
    )
    # add colorbar
    cbar = add_colorbar(im, aspect=20, pad_fraction=0.5)
    ticks = list(np.linspace(vmin, vmax, 6))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{i:0.2f}" for i in ticks])
    cbar.ax.tick_params(labelsize=visual_settings_dict["colorbar_font"])

    # ######################### STD gradCAM background
    to_plot = gradCAM_dictionary["gradCAM_raw"][
        :, :, :, background_class, layer_of_interest
    ].squeeze()

    if scale:
        vmin = 0
        vmax = 1
        # normalize grad cam
        to_plot = (to_plot - to_plot.min()) / (to_plot.max() - to_plot.min())
    else:
        vmin = to_plot.min()
        vmax = to_plot.max()

    if len(to_plot.shape) != 2:
        to_plot = np.fliplr(np.rot90(to_plot.std(axis=0), k=3))
    else:
        to_plot = np.fliplr(np.rot90(to_plot, k=3))

    aus_ax = ax[1, 3]
    im = aus_ax.imshow(
        to_plot,
        cmap=visual_settings_dict["cmap_STD_gradCAM"],
        interpolation=None,
        vmin=vmin,
        vmax=vmax,
    )
    aus_ax.axis("off")
    aus_ax.set_title(
        f"GradCAM STD for background (# {background_class}, layer nbr. {layer_of_interest})",
        fontsize=visual_settings_dict["title_font_size"],
    )
    # add colorbar
    cbar = add_colorbar(im, aspect=20, pad_fraction=0.5)
    ticks = list(np.linspace(vmin, vmax, 6))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{i:0.2f}" for i in ticks])
    cbar.ax.tick_params(labelsize=visual_settings_dict["colorbar_font"])

    # ###################################### Write information infromation
    """
    Write statistical information in the axis
    - nbr samples in the two distributions
    - Attribution threshold
    - % significant in brain region
    - % significant in tumor region
    """
    aus_ax = ax[0, 4]

    aus_ax.text(
        0.1,
        visual_settings_dict["start_upper_row"],
        f"Nbr. samples in each distribution: {gradCAM_dictionary['original_image_prediction_distribution'].shape[0]}",
        transform=aus_ax.transAxes,
        size="xx-large",
        color="k",
        horizontalalignment="left",
        verticalalignment="center",
    )
    aus_ax.text(
        0.1,
        visual_settings_dict["start_upper_row"]
        - visual_settings_dict["space_between_rows"] * 1,
        f"Attribution threshold: {attribution_thr}",
        transform=aus_ax.transAxes,
        size="xx-large",
        color="k",
        horizontalalignment="left",
        verticalalignment="center",
    )

    brain_mask = (anatomical_img != 0).astype(int)

    pr_in_brain = np.sum(brain_mask * (gradCam_mean_map >= attribution_thr)) / np.sum(
        gradCam_mean_map >= attribution_thr
    )
    aus_ax.text(
        0.1,
        visual_settings_dict["start_upper_row"]
        - visual_settings_dict["space_between_rows"] * 2,
        f"Fraction gradCAM values above attribution threshold within the brain region: {pr_in_brain:0.3f}",
        transform=aus_ax.transAxes,
        size="xx-large",
        color="k",
        horizontalalignment="left",
        verticalalignment="center",
    )
    if any(
        [
            plot_annotation_countour,
            gt == 1,
            "focus_metrics" in list(gradCAM_dictionary.keys()),
        ]
    ):
        pr_in_tumor = np.sum(
            annotation * (gradCam_mean_map >= attribution_thr)
        ) / np.sum(gradCam_mean_map >= attribution_thr)
    else:
        pr_in_tumor = np.nan

    aus_ax.text(
        0.1,
        visual_settings_dict["start_upper_row"]
        - visual_settings_dict["space_between_rows"] * 3,
        f"Fraction of gradCAM above attribution threshold within the tumor region: {pr_in_tumor:0.3f}",
        transform=aus_ax.transAxes,
        size="xx-large",
        color="k",
        horizontalalignment="left",
        verticalalignment="center",
    )
    pr_in_outside = 1 - pr_in_brain
    aus_ax.text(
        0.1,
        visual_settings_dict["start_upper_row"]
        - visual_settings_dict["space_between_rows"] * 4,
        f"Fraction of gradCAM above attribution threshold outside the brain region: {pr_in_outside:0.3f}",
        transform=aus_ax.transAxes,
        size="xx-large",
        color="k",
        horizontalalignment="left",
        verticalalignment="center",
    )
    aus_ax.axis("off")
    ax[1, 0].axis("off")
    fig.add_subplot(aus_ax)

    # ########################### plot brain (and annotation coutour)
    aus_ax = ax[1, 4]
    aus_ax.axis("off")

    to_plot = gradCam_mean_map >= attribution_thr
    if scale:
        vmin = 0
        vmax = 1
    else:
        vmin = to_plot.min()
        vmax = to_plot.max()

    im = aus_ax.imshow(
        to_plot,
        cmap="gray",
        interpolation=None,
        vmin=vmin,
        vmax=vmax,
    )
    aus_ax.axis("off")
    aus_ax.set_title(
        f"Thresholded mean attribution map (thr={attribution_thr})",
        fontsize=visual_settings_dict["title_font_size"],
    )
    # add colorbar
    cbar = add_colorbar(im, aspect=20, pad_fraction=0.5)
    ticks = list(np.linspace(vmin, vmax, 6))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{i:0.2f}" for i in ticks])
    cbar.ax.tick_params(labelsize=visual_settings_dict["colorbar_font"])

    # add annotation
    if plot_annotation_countour:
        if any([np.max(gt) == 1, "focus_metrics" in list(gradCAM_dict.keys())]):
            # plot countour over the image
            aus_ax.imshow(
                countour, cmap=visual_settings_dict["cmap_annotation_contour"]
            )

    # add brain countour
    brain_mask_contour = brain_mask - ndimage.binary_erosion(brain_mask, iterations=3)
    brain_mask_contour = np.ma.masked_where(brain_mask_contour == 0, brain_mask_contour)
    # plot countour over the image
    aus_ax.imshow(brain_mask_contour, cmap=visual_settings_dict["cmap_brain_contour"])

    fig.add_subplot(aus_ax)

    if save_path:
        fig.savefig(
            save_path + ".png",
            bbox_inches="tight",
            dpi=visual_settings_dict["dpi"],
            facecolor=fig.get_facecolor(),
            transparent=True,
        )
        fig.savefig(
            save_path + ".pdf",
            bbox_inches="tight",
            dpi=visual_settings_dict["dpi"],
            facecolor=fig.get_facecolor(),
            transparent=True,
        )
        if draw == True:
            plt.show()
        else:
            plt.close(fig)


#%% DEFINE PATHS
to_print = "    Plot of GradCAMS (from saved .npy files)   "

print(f'\n{"-"*len(to_print)}')
print(to_print)
print(f'{"-"*len(to_print)}\n')

su_debug_flag = True

# --------------------------------------
# read the input arguments and set the base folder
# --------------------------------------
if not su_debug_flag:
    parser = argparse.ArgumentParser(
        description="Run cross validation training on a combination of MRI modalities."
    )
    parser.add_argument(
        "-ofp",
        "--GRADCAM_FILES_PATH",
        required=True,
        help="Provide the path to where the gradCAM.npy files fro each image are saved.",
    )

    args_dict = dict(vars(parser.parse_args()))

else:
    # # # # # # # # # # # # # # DEBUG
    print("Running in debug mode.")
    args_dict = {
        "GRADCAM_FILES_PATH": "/flush/iulta54/Research/P5-MICCAI2023/trained_models_archive/Detection_infra_optm_ADAM_EfficientNet_TFRdata_True_modality_T2_loss_MCC_and_CCE_Loss_lr_0.001_batchSize_32_pretrained_True_useAge_False_useGradCAM_False/Explainability_analysis/GradCAMs",
    }
# specify where to save the results
args_dict["SAVE_PATH"] = os.path.join(args_dict["GRADCAM_FILES_PATH"], "Visualization")
Path(args_dict["SAVE_PATH"]).mkdir(parents=True, exist_ok=True)

# print input variables
max_len = max([len(key) for key in args_dict])
[print(f"{key:{max_len}s}: {value}") for key, value in args_dict.items()]

# get the files to analyse
args_dict["gradCAM_FILES"] = glob.glob(
    os.path.join(args_dict["GRADCAM_FILES_PATH"], "*.npy")
)
print(os.path.join(args_dict["GRADCAM_FILES_PATH"], "*.npy"))
print(f"Found {len(args_dict['gradCAM_FILES'])} gradCAM files to work on.")

#%% PLOT MAPS
print("Comparing the original prediction distribution with the pathed one.")

import cmasher as cmr

visual_settings_dict = {
    "title_font_size": 20,
    "colorbar_font": 15,
    "cmap_mean_gradCAM": plt.get_cmap("cmr.torch"),
    "cmap_STD_gradCAM": plt.get_cmap("cmr.torch"),
    "cmap_annotation_contour": "hsv",
    "cmap_brain_contour": "Set2",
    "space_between_rows": 0.05,
    "start_upper_row": 0.7,
    "dpi": 100,
}

for idx, f in enumerate(args_dict["gradCAM_FILES"]):
    print(f"Working on image {idx+1}/{len(args_dict['gradCAM_FILES'])}\r", end="")
    # open .npy file
    gradCAM_dict = np.load(f, allow_pickle=True).item()

    # get p-value for each region in the image
    interest_class = 1
    layers_of_interest = range(81)

    for layer_of_interest in layers_of_interest:
        save_path = os.path.join(
            args_dict["SAVE_PATH"],
            f"result_analysis_class_of_interest_{interest_class}_layer_{layer_of_interest}_{Path(f).stem}",
        )
        plot_gradCAM(
            gradCAM_dict,
            class_of_interest=interest_class,
            layer_of_interest=layer_of_interest,
            attribution_thr=0.6,
            save_path=save_path,
            scale=True,
            draw=False,
            plot_annotation_countour=True,
            visual_settings_dict=visual_settings_dict,
        )

# %%
