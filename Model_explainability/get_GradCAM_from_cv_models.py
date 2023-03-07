#%%
"""
Script that computes the GradCAM values for the models trained trhough cross validation.
The script ONLY computes and saves the values. It does not perform statistical comparison between the different
populations or computes other metrics. This is carried out by a separate script.

STEPS
1 - load data on which to perform the computation (as for now using the data generator
    to get he files to work with. In the future one could load the images one at the time).
2 - for all the models
    ¤ build gradCAM object
    ¤ get the GradCAM map for all the images
    ¤ save the values in a dictionary
3 - save the dictionary to a file where the script for the analysis of the prediction disctibution can work with

The dictionary has the following fields:
- anatomical_image : np.array
    This is the anatomical image (the first channel if multi-channel) used for plotting
- original_image_prediction_distribution : np.array
    Array of size [N, C] containing the logits of every model (N) for all the classes (C)
- gradCAM : np.array
    Array of size [W, H, C, L, N] containing the GradCAM image (W, H)
    for evey class (C), every conv. layer L and every model (N)
"""
from calendar import c
from itertools import count
import os
import cv2
import sys
import json
import glob
import time
import types
import pathlib
import random
import argparse
import importlib
import scipy
import numpy as np
import pandas as pd
from pathlib import Path
from random import shuffle
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import matplotlib as mpl
import logging

from PIL import Image

# local imports
# local imports
import utilities
import data_utilities


#%% DEFINE UTILITIES
# define GradCAM object
"""
Grad-CAM implementation [1] as described in post available at [2].
[1] Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-cam:
    Visual explanations from deep networks via gradient-based localization.
    InProceedings of the IEEE international conference on computer vision 2017
    (pp. 618-626).
[2] https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
"""


class gradCAM:
    def __init__(
        self,
        model,
        classIdx,
        layerName=None,
        use_image_prediction=True,
        ViT=False,
        is_3D=False,
        debug=False,
    ):
        """
        model: model to inspect
        classIdx: index of the class to ispect
        layerName: which layer to visualize
        """
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        self.debug = debug
        self.use_image_prediction = use_image_prediction
        self.is_ViT = ViT
        self.is_3D = is_3D
        self.is_efficientNet = is_efficientNet

        # if the layerName is not provided, find the last conv layer in the model
        if all([self.layerName is None, not is_efficientNet]):
            self.layerName = self.find_target_layer()
        else:
            if self.debug is True:
                print(
                    "GradCAM - using layer {}".format(
                        self.model.get_layer(self.layerName).name
                    )
                )

    def find_target_layer(self):
        """
        Finds the last convolutional layer in the model by looping throught the
        available layers
        """
        for layer in reversed(self.model.layers):
            # check if it is a 2D conv layer (which means that needs to have
            # 4 dimensions [batch, width, hight, channels])
            if len(layer.output_shape) == 4:
                # check that is a conv layer
                if layer.name.find("conv") != -1:
                    if self.debug is True:
                        print("GradCAM - using layer {}".format(layer.name))
                    return layer.name

        if self.layerName is None:
            # if no convolutional layer have been found, rase an error since
            # Grad-CAM can not work
            raise ValueError("Could not find a 4D layer. Cannot apply GradCAM")

    def compute_heatmap(self, image, eps=1e-6):
        """
        Compute the L_grad-cam^c as defined in the original article, that is the
        weighted sum over feature maps in the given layer with weights based on
        the importance of the feature map on the classsification on the inspected
        class.
        This is done by supplying
        1 - an input to the pre-trained model
        2 - the output of the selected conv layer
        3 - the final softmax activation of the model
        """
        # this is a gradient model that we will use to obtain the gradients from
        # with respect to an image to construct the heatmaps
        # if self.is_efficientNet:
        #     # this is convoluted since the pretrained model is nested. Ugly but works
        #     pre_model = tf.keras.models.Model(
        #         self.model.input, self.model.layers[0].output
        #     )
        #     eff_net = tf.keras.models.Model(
        #         self.model.layers[1].input,
        #         self.model.layers[1].get_layer(self.layerName).output,
        #     )
        #     gradModel = tf.keras.models.Model(
        #         self.model.input, [eff_net.call(pre_model.output), self.model.output]
        #     )
        # else:
        gradModel = tf.keras.Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer(self.layerName).output,
                self.model.output,
            ],
        )

        # replacing softmax with linear activation
        gradModel.layers[-1].activation = tf.keras.activations.linear

        if self.debug is True:
            gradModel.summary()

        # use the tensorflow gradient tape to store the gradients
        with tf.GradientTape() as tape:
            """
            cast the image tensor to a float-32 data type, pass the
            image through the gradient model, and grab the loss
            associated with the specific class index.
            """
            inputs = tf.cast(image, tf.float32)
            tape.watch(inputs)
            (convOutputs, predictions) = gradModel(inputs)
            # check if the prediction is a list (VAE)
            if type(predictions) is list:
                # the model is a VEA, taking only the prediction
                predictions = predictions[4]
            pred = tf.argmax(predictions, axis=1)
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, convOutputs)

        # sometimes grads becomes NoneType
        if grads is None:
            grads = tf.zeros_like(convOutputs)
        """
        compute the guided gradients.
         - positive gradients if the classIdx matches the prediction (I want to
            know which values make the probability of that class to be high)
         - negative gradients if the classIdx != the predicted class (I want to
            know which gradients pushed down the probability for that class)
        """
        if self.use_image_prediction == True:
            if self.classIdx == pred:
                castConvOutputs = tf.cast(convOutputs > 0, tf.float32)
                castGrads = tf.cast(grads > 0, tf.float32)
            else:
                castConvOutputs = tf.cast(convOutputs <= 0, tf.float32)
                castGrads = tf.cast(grads <= 0, tf.float32)
        else:
            castConvOutputs = tf.cast(convOutputs > 0, tf.float32)
            castGrads = tf.cast(grads > 0, tf.float32)
        guidedGrads = castConvOutputs * castGrads * grads

        # remove the batch dimension
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the weight value for each feature map in the conv layer based
        # on the guided gradient
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # now that we have the activation map for the specific layer, we need
        # to resize it to be the same as the input image
        if self.is_ViT:
            if self.is_3D:
                # here we take the middle slice (don't take mean or sum since the
                # channels are not conv filters, but the actual activation for
                # the different images in the sequence). This is different compared
                # to a normal conv3d, where the chanlles are descriptive of all
                # the images at the same time
                dim = int(np.sqrt(cam.shape[0] / image.shape[3]))
                (w, h) = (image.shape[2], image.shape[1])
                heatmap = cam.numpy().reshape((dim, dim, image.shape[3]))
                heatmap = heatmap[:, :, heatmap.shape[-1] // 2]
                heatmap = cv2.resize(heatmap, (w, h))
            else:
                dim = int(np.sqrt(cam.shape[0]))
                (w, h) = (image.shape[2], image.shape[1])
                heatmap = cam.numpy().reshape((dim, dim))
                heatmap = cv2.resize(heatmap, (w, h))
        else:
            if self.is_3D:
                # reshape cam to the layer input shape and then take the middle
                # slice
                layer_shape = self.model.get_layer(self.layerName).input_shape
                heatmap = cam.numpy().reshape(
                    (layer_shape[1], layer_shape[2], layer_shape[3])
                )
                heatmap = np.mean(heatmap, axis=-1)
                # heatmap = heatmap[:,:,heatmap.shape[-1]//2]
                (w, h) = (image.shape[2], image.shape[1])
                heatmap = cv2.resize(heatmap, (w, h))
            else:
                (w, h) = (image.shape[2], image.shape[1])
                heatmap = cv2.resize(cam.numpy(), (w, h))

        # normalize teh heat map in [0,1] and rescale to [0, 255]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap_raw = (heatmap * 255).astype("uint8")

        # create heatmap based ont he colormap setting
        heatmap_rgb = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_VIRIDIS).astype(
            "float32"
        )

        return heatmap_raw, heatmap_rgb

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS):

        # create heatmap based ont he colormap setting
        heatmap = cv2.applyColorMap(heatmap, colormap).astype("float32")

        if image.shape[-1] == 1:
            # convert image from grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype("float32")

        output = cv2.addWeighted(image, alpha, heatmap, (1 - alpha), 0)

        # return both the heatmap and the overlayed output
        return (heatmap, output)


#%% GET VARIABLES
to_print = "    Get GradCAM MAPS script from CROSS VALIDATION MODELS   "

print(f'\n{"-"*len(to_print)}')
print(to_print)
print(f'{"-"*len(to_print)}\n')

su_debug_flag = True

# --------------------------------------
# read the input arguments and set the base folder
# --------------------------------------
if not su_debug_flag:
    parser = argparse.ArgumentParser(
        description="Run GradCAM computation for models trained through cross validation."
    )
    parser.add_argument(
        "-ptm",
        "--PATH_TO_MODEL",
        required=True,
        help="Provide the path to where the cross validation repetition folders for the desired run are present.",
    )
    parser.add_argument(
        "-df",
        "--IMG_DATASET_FOLDER",
        required=True,
        type=str,
        help="Provide the Image Dataset Folder where the the images to run the gradCAM are located",
    )
    parser.add_argument(
        "-n",
        "--NBR_IMG_TO_PROCESS",
        required=False,
        help="Specify the number of images to process Default 10.",
        default=10,
    )
    parser.add_argument(
        "-g",
        "--GPU",
        required=False,
        help="Specify the GPU number to use.",
        default=0,
    )

    args_dict = dict(vars(parser.parse_args()))
    # bring variable to the right format
    args_dict["NBR_IMG_TO_PROCESS"] = int(args_dict["NBR_IMG_TO_PROCESS"])


else:
    # # # # # # # # # # # # # # DEBUG
    print("Running in debug mode.")
    args_dict = {
        "PATH_TO_MODELS": "/flush/iulta54/Research/P5-MICCAI2023/trained_models_archive/Detection_infra_optm_ADAM_EfficientNet_TFRdata_True_modality_T2_loss_MCC_and_CCE_Loss_lr_0.001_batchSize_32_pretrained_True_useAge_False_useGradCAM_False",
        "IMG_DATASET_FOLDER": "/flush/iulta54/Research/Data/CBTN/EXTRACTED_SLICES/T2",
        "ANNOTATION_DATASET_FOLDER": "/flush/iulta54/Research/Data/CBTN/EXTRACTED_SLICES/T2",
        "NBR_IMG_TO_PROCESS": 3,
        "GPU": "0",
    }

# --------------------------------------
# set GPU
# --------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args_dict["GPU"]
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import warnings

tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.layers as layers
import keras

devices = tf.config.list_physical_devices("GPU")

if devices:
    print(f'Running training on GPU # {args_dict["GPU"]} \n')
    warnings.simplefilter(action="ignore", category=FutureWarning)
    tf.config.experimental.set_memory_growth(devices[0], True)
else:
    Warning(
        f"ATTENTION!!! MODEL RUNNING ON CPU. Check implementation in case GPU is wanted."
    )

# get path to the repetition folders
args_dict["CV_REPETITION_FOLDERS"] = glob.glob(
    os.path.join(args_dict["PATH_TO_MODELS"], "fold_*")
)
# define path where to save the occlusion distributions
args_dict["SAVE_PATH"] = os.path.join(
    args_dict["PATH_TO_MODELS"],
    "Explainability_analysis",
    f"GradCAMs",
)
Path(args_dict["SAVE_PATH"]).mkdir(parents=True, exist_ok=True)

# print input variables
max_len = max([len(key) for key in args_dict])
[print(f"{key:{max_len}s}: {value}") for key, value in args_dict.items()]

# %% CREATE DATA GENERATOR (THIS SHOULD BE CHANGED BASED ON HOW YOUR GENERATOR or DATA LOADING PROCESS WORKS)
print(f"Loading data to work on...")
importlib.reload(data_utilities)
"""
Note that in the case of repeated cross validations, each repetition has its own
test samples. Thus, here, we load all the available (the one used by all the
repetitions for training, validation and testing) saving it such that each repetitions
can select the right samples to use.

Saving the data as np arrays, where the index in the array, e.g. 1, points to the
data for subject 1.
"""

with open(
    os.path.join(
        os.path.dirname(args_dict["CV_REPETITION_FOLDERS"][0]),
        "train_val_test_files.json",
    )
) as file:
    config = json.load(file)
    img_files = [
        os.path.join(args_dict["IMG_DATASET_FOLDER"], f"{Path(f).stem}.png")
        for f in config["test"]
    ]
    # remove infra supra from the file name (not in the .png images)
    aus_img_files = []
    img_files = [
        os.path.join(
            os.path.dirname(f),
            "_".join(
                [
                    os.path.basename(f).split("_")[i]
                    for i in range(len(os.path.basename(f).split("_")))
                    if i != 1
                ]
            ),
        )
        for f in img_files
    ]

    # remove images that are not available
    # img_files = [f for f in img_files if os.path.isfile(f)]
    # only take args_dict['NBR_IMG_TO_PROCESS'] random images
    random.shuffle(img_files)
    # just for debug take images that have tumor
    img_files = [f for f in img_files if "label_1" in os.path.basename(f)]
    img_files = img_files[0 : args_dict["NBR_IMG_TO_PROCESS"]]

# ###################
# use all the images in the given dataset path
# img_files = glob.glob(os.path.join(args_dict["IMG_DATASET_FOLDER"], "*.png"))
# remove annotation files
# img_files = [f for f in img_files if "annotation" not in os.path.basename(f)]
# ###################

target_size = (224, 224)
img_gen = data_utilities.img_data_generator(
    sample_files=img_files,
    target_size=target_size,
    batch_size=1,
    dataset_type="testing",
    normalize_img=False,
)

# build dictionary where the images to work on are located
IMAGES = {}

for idx, ((x, y), f) in enumerate(zip(img_gen, img_gen.filenames)):
    IMAGES[idx] = {"image_file_name": f, "image": x, "label": y[0]}
    # if the label is 1 (there is tumor) get the annotation file name and load the annotation
    if y[0].argmax() == 1:
        # get annotation file

        # BRATS VERSION
        # annotation_file = f"{'_'.join(Path(f).stem.split('_')[0:3])}_{'_'.join(Path(f).stem.split('_')[4::])}_annotation.jpeg"

        # CBTN VERSION
        annotation_file = f"_".join([Path(f).stem, "annotation.png"])

        annotation_file = os.path.join(
            args_dict["ANNOTATION_DATASET_FOLDER"], annotation_file
        )
        # check that the file exists. If so, load the image
        if os.path.isfile(annotation_file):
            annotation_image = Image.open(annotation_file).convert("L")
            # resize to target_size
            annotation_image = annotation_image.resize(target_size)
            # make binary
            annotation_image = (np.array(annotation_image) >= 30).astype(int)

            IMAGES[idx]["annotation"] = annotation_image
    else:
        IMAGES[idx]["annotation"] = np.zeros_like(np.squeeze(x))

    if idx + 1 == len(img_gen):
        break

print(f"Done!")
print(f"Pool of test data to work on: {len(img_gen)}")

#%% STEP 2 - GET GradCAMs FROM ALL THE MODELS
# define parameters
dataset_type = "test"
model_version = "last"
is_efficientNet = True

# get models to run prediction on (along with the index of the images to use for
# each of them based on the dataset).
MODELS = []
for f in args_dict["CV_REPETITION_FOLDERS"]:
    # fold_index
    f_indx = int(os.path.basename(f).split("_")[-1]) - 1
    # get index of subject for this fold
    if dataset_type == "test":
        aus_subjects = config[dataset_type]
        # ################################
        aus_subjects.extend(config["train"][1])
        aus_subjects.extend(config["validation"][1])
        # ################################
    else:
        aus_subjects = config[dataset_type][f_indx]
    # get subject index from the names
    aus_subjects = [int(s.split(".")[0].split("_")[-1]) for s in aus_subjects]
    # save information
    aus_dict = {
        "model_path": os.path.join(f, "last_model", "last_model"),
        "indx_subjects": aus_subjects,
    }
    if model_version == "best":
        aus_dict["best_weights_path"] = os.path.join(f, "best_model_weights", "")
    MODELS.append(aus_dict)

"""
Every cross validation run is tested on a different pool of test subjects. Thus,
not all the models share the same test images. To make sure that the occlusion
distributions for every image are obtained only from the models for which the
image is in the test set, the data is save as follows
[subject_index][img_index]: {'original_image_prediction_distribution':[] (stores the values from the models that are tested on this image)}
                            {'occluded_image_prediction_distribution':[] (stores the values from the models that are tested on this image)}
THe loops are as follows
for every model:
    - get test subject index
    for all the test subjects:
        for all the images in a test subject:
            - get model prediction on original image
            - get occlusion map on accluded image
"""
# build where to save the results
results = dict.fromkeys(IMAGES.keys())
for img in results.keys():
    results[img] = {
        "original_image_prediction_distribution": [],
        "gradCAM_raw": [],
        "image_file_name": [],
        "ground_truth": [],
    }

for m_idx, model_dict in enumerate(MODELS):
    # load model
    if os.path.exists(model_dict["model_path"]):
        model = tf.keras.models.load_model(model_dict["model_path"], compile=False)
    else:
        raise Exception("Model not found")
    # load best model if needed
    if model_version == "best":
        try:
            model.load_weights(model_dict["best_weights_path"])
        except:
            raise Exception(
                f'Weights of best model not found! Give {model_dict["best_weights_path"]}'
            )
    # deactivate augmentation layer if present
    # find index of Augmentation layer
    idy = [i for i, l in enumerate(model.layers) if l.name == "Augmentation"]
    if idy:
        idy = idy[0]
        model.layers[idy].layers[0].horizontal = False
        model.layers[idy].layers[0].vertical = False
        model.layers[idy].layers[1].lower = 0
        model.layers[idy].layers[1].upper = 0

    """
    This heuristic need to change based on the model architecture as well as layer names.
    Check model.summary() to get an idea of the different names.
    """
    name_layers = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.convolutional.Conv2D):
            name_layers.append(layer.name)
    # name_layers = name_layers[0:10]

    break
    # %%
    # loop through the test images of this model
    for img_idx in range(len(IMAGES)):
        image = IMAGES[img_idx]["image"]
        if np.any(image != -1):
            # save original image classification
            results[img_idx]["original_image_prediction_distribution"].append(
                model(image).numpy()[0]
            )
            # save file name
            results[img_idx]["image_file_name"] = IMAGES[img_idx]["image_file_name"]
            # save gt
            results[img_idx]["ground_truth"] = IMAGES[img_idx]["label"]
            # build gradCAM object for each layer
            temp_l_raw = []
            for idx_l, nl in enumerate(name_layers):
                # ausiliary for gradCAM for each layer
                temp_c_raw = []
                for idx_c, c in enumerate([0, 1]):
                    print(
                        f"Working on model {m_idx+1}/{len(MODELS)} (image {img_idx+1:{len(str(len(img_gen)))}}/{len(img_gen)}, layer {idx_l+1}/{len(name_layers)}, class {idx_c+1}/{len([0,1])}) \r",
                        end="",
                    )
                    # define gradCAM object for this layer and channel
                    cam = gradCAM(model, c, layerName=nl)
                    # compute gradCAM
                    aus_raw, _ = cam.compute_heatmap(image)
                    # save infom for this channel
                    temp_c_raw.append(aus_raw)
                # save information for this
                temp_l_raw.append(temp_c_raw)
            # save information
            results[img_idx]["gradCAM_raw"].append(temp_l_raw)

# %% SAVE RESULTS - COLLECT INFORMATION FROM EVERY IMAGE AND SAVE DATA FOR EVERY IMAGE INDEPENDENTLY
"""
- anatomical_image : [W,H,nCh]
- original_image_prediction_distribution : [N,C]
- gradCAM_raw : [N,W,H,Ch,L]
"""

for img_idx in results.keys():
    # the image has been processed, save
    print(
        f"Saving information for img {img_idx+1:{len(str(len(img_gen)))}}/{len(img_gen)}"
    )
    aus_original_image_prediction_distribution = np.array(
        results[img_idx]["original_image_prediction_distribution"]
    )
    aus_occluded_image_prediction_distribution = []

    # pack information in a dict
    aus_dict = {
        "ground_truth": results[img_idx]["ground_truth"],
        "anatomical_image": np.squeeze(IMAGES[img_idx]["image"]),
        "annotation_image": IMAGES[img_idx]["annotation"],
        "anatomical_image_modality": "None",
        "original_image_prediction_distribution": np.array(
            results[img_idx]["original_image_prediction_distribution"]
        ),
        "gradCAM_raw": np.array(results[img_idx]["gradCAM_raw"]).transpose(
            0, 3, 4, 2, 1
        ),
        "layer_names": name_layers,
    }
    np.save(
        os.path.join(
            args_dict["SAVE_PATH"],
            f"{Path(results[img_idx]['image_file_name']).stem}.npy",
        ),
        aus_dict,
    )

# %%
