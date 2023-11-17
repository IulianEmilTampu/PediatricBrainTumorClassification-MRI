# %% 
'''
Script that given a dataset split originated using the main_CBTN_v1_dataset.py script, 
performs a spectral analysis of the different sets.
'''

import os
import pandas as pd
import pathlib
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms as T
from datetime import datetime
import copy
from PIL import Image
from torchvision.utils import make_grid
import scipy.signal as signal

import dataset_utilities

# %% UTILITIES

def plot_grid_from_list(list_of_images, vmin:float=None, vmax:float=None, draw:bool=True, save_path:str=None, prefix:str='Image_grid'):
    from mpl_toolkits.axes_grid1 import ImageGrid

    # get appropriate number of roes and columns (all images should be plotted)
    nbr_rows_cols = int(np.sqrt(len(list_of_images)))
    if nbr_rows_cols ** 2 < len(list_of_images):
        nbr_rows_cols += 1

    fig = plt.figure(figsize=(15, 15))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(nbr_rows_cols, nbr_rows_cols),
                    axes_pad=0.02,
                    label_mode="L",
                    )
    for ax, im in zip(grid, list_of_images):
        ax.imshow(im, origin="lower", cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_xticks([]), ax.set_yticks([])
    
    if save_path:
        # save image
        fig.savefig(os.path.join(save_path, f'{prefix}.jpeg'), bbox_inches='tight', dpi = 100)
    if draw:
        plt.show()
    else:
        plt.close(fig)

def gkern(kernlen=224, std=50):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

# %% LOAD DATASET
preprocess = T.Compose(
    [
        T.Resize(size=(224,224), antialias=True),
        T.ToTensor(),
    ],
)

DATASET_PATH = "/flush2/iulta54/Data/CBTN_v1/FILTERED_SLICES_RLP_20_80"
PATH_TO_SPLIT_DF =  '/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/trained_model_archive/TESTs_20231116/ResNet50_pretrained_True_frozen_True_0.6_LR_0.0001_BATCH_128_AUGMENTATION_True_OPTIM_adam_SCHEDULER_exponential_MLPNODES_512_t145005/REPETITION_1/data_split_information.csv'
SET = 'test'
SAVE_PATH = os.path.join(os.getcwd(), 'spectral_analysis', f'TESTs_{datetime.now().strftime("%y%m%d")}', SET.capitalize())
pathlib.Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
dataset_split_df = pd.read_csv(PATH_TO_SPLIT_DF, low_memory=False)


set_file_paths = list(dataset_split_df.loc[
                    dataset_split_df[f"fold_1"] == SET
                ]["file_path"])
set_files_labels = list(dataset_split_df.loc[
                    dataset_split_df[f"fold_1"] == SET
                ]["target"])

print(f"Testing...(testing on {len(set_file_paths)} files)")


# dataset = dataset_utilities.PNGDatasetFromFolder(item_list=set_file_paths, labels=set_files_labels, transform=preprocess)
# %% FOR EVERY SUBJECT, GET THE CENTRAL SLICE AND PLOT 2D FFT
unique_subjects = list(pd.unique(dataset_split_df.loc[
                    dataset_split_df[f"fold_1"] == SET
                ]["subject_IDs"]))

rlp_to_investigate = 50
raw_imgs = []
ffts = []
for subject in unique_subjects:
    aus_df = copy.copy(dataset_split_df.loc[(
                    dataset_split_df[f"fold_1"] == SET) & (dataset_split_df[f"subject_IDs"] == subject)
                ])
    # get the slice closer to the specified relative position
    aus_df['difference'] = abs(aus_df['tumor_relative_position'] - rlp_to_investigate)
    index_selected_slice = aus_df['difference'].idxmin()
    
    # make a dataset using only this slice
    slice_path = [aus_df.loc[index_selected_slice, 'file_path']]
    slice_target = [aus_df.loc[index_selected_slice, 'target']]

    # make dataset to emulate dataloader
    aus_dataset = dataset_utilities.PNGDatasetFromFolder(item_list=slice_path, labels=slice_target, transform=None)

    # take out image compute FFT
    img, label = next(iter(aus_dataset))
    img = np.rot90(img.numpy().transpose(1,2,0), k=1)[:,:,0]

    # from https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    # save for later
    raw_imgs.append(img)
    ffts.append(magnitude_spectrum)

# %% plot using image grid

quantile_max = 0.98
quantile_min = 1 - quantile_max
plot_grid_from_list(raw_imgs,vmin=np.quantile(raw_imgs, quantile_min),vmax=np.quantile(raw_imgs, quantile_max), draw=False, save_path=SAVE_PATH, prefix='Raw_images')
plot_grid_from_list(ffts, draw=False, vmin=np.quantile(ffts, quantile_min),vmax=np.quantile(ffts, quantile_max), save_path=SAVE_PATH, prefix='Magnitude_images')

# get vmin and vmax to be use for plotting all the slices for all the set images
raw_img_vmin, raw_img_vmax = np.quantile(raw_imgs, quantile_min), np.quantile(raw_imgs, quantile_max)
magnitude_img_vmin, magnitude_img_vmax = np.quantile(ffts, quantile_min), np.quantile(ffts, quantile_max)

# %% FOR EVERY SUBJECT, PLOT ALL THE SLICES IN SEQUENCE
for idx, subject in enumerate(unique_subjects):
    # print progress
    print(f'Working on subject {idx+1} of {len(unique_subjects)}')
    # where to save the images for this subject
    subject_raw_imgs = []
    subject_magnitude_imgs = []

    # take out data from the dataframe
    aus_df = copy.copy(dataset_split_df.loc[(
                    dataset_split_df[f"fold_1"] == SET) & (dataset_split_df[f"subject_IDs"] == subject)
                ])

    slice_path = list(aus_df['file_path'])
    slice_target = list(aus_df['target'])

    # make dataset to emulate dataloader
    aus_dataset = dataset_utilities.PNGDatasetFromFolder(item_list=slice_path, labels=slice_target, transform=preprocess)

    # take out image and compute FFT
    aus_dataset = iter(aus_dataset)
    for i in range(len(slice_path)):
        img, label = next(aus_dataset)
        img = np.rot90(img.numpy().transpose(1,2,0), k=1)[:,:,0]

        # from https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))

        # save for later
        subject_raw_imgs.append(img)
        subject_magnitude_imgs.append(magnitude_spectrum)
    
    # plot as a grid
    plot_grid_from_list(subject_raw_imgs, vmin=raw_img_vmin,vmax=raw_img_vmax,draw=False, save_path=SAVE_PATH, prefix=f'Raw_images_{subject}')
    plot_grid_from_list(subject_magnitude_imgs, vmin=magnitude_img_vmin,vmax=magnitude_img_vmax, draw=False, save_path=SAVE_PATH, prefix=f'Magnitude_images_{subject}')

# %% APPLY LOW-PASS FILTER TO THE IMAGES (TRYING TO FIX UNEVENNES IN THE FFT)

def gkern(kernlen=224, std=50):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

LP_cut_off = 0.2 # as ratio of frequences to keep

for idx, subject in enumerate(unique_subjects):
    # print progress
    print(f'Working on subject {idx+1} of {len(unique_subjects)}')
    # where to save the images for this subject
    subject_LP_imgs = []
    subject_magnitude_imgs = []

    # take out data from the dataframe
    aus_df = copy.copy(dataset_split_df.loc[(
                    dataset_split_df[f"fold_1"] == SET) & (dataset_split_df[f"subject_IDs"] == subject)
                ])

    slice_path = list(aus_df['file_path'])
    slice_target = list(aus_df['target'])

    # make dataset to emulate dataloader
    aus_dataset = dataset_utilities.PNGDatasetFromFolder(item_list=slice_path, labels=slice_target, transform=preprocess)

    # take out image and compute FFT
    aus_dataset = iter(aus_dataset)
    for i in range(len(slice_path)):
        img, label = next(aus_dataset)
        img = np.rot90(img.numpy().transpose(1,2,0), k=1)[:,:,0]

        # from https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        raw_magnitude = 20*np.log(np.abs(fshift))

        # build gaussian filter and apply
        rows, cols = img.shape
        crow,ccol = int(rows/2) , int(cols/2)
        gfilter = gkern(std=rows*LP_cut_off)

        # mask = np.zeros((rows,cols))
        # mask[crow-LP_cut_off:crow+LP_cut_off, ccol-LP_cut_off:ccol+LP_cut_off] = 1

        fshift = fshift * gfilter
    
        f_ishift = np.fft.ifftshift(fshift)
        LP_img = np.fft.ifft2(f_ishift)
        LP_img = np.abs(LP_img)

        # get the FFT of this new image
        f = np.fft.fft2(LP_img)
        fshift = np.fft.fftshift(f)
        LP_magnitude = 20*np.log(np.abs(fshift))

        # save for later
        subject_LP_imgs.append(LP_img)
        subject_magnitude_imgs.append(LP_magnitude)
    
    # plot as a grid
    plot_grid_from_list(subject_LP_imgs, vmin=raw_img_vmin,vmax=raw_img_vmax,draw=False, save_path=SAVE_PATH, prefix=f'LP_{LP_cut_off:0.2f}_raw_images_{subject}')
    plot_grid_from_list(subject_magnitude_imgs, vmin=magnitude_img_vmin,vmax=magnitude_img_vmax, draw=False, save_path=SAVE_PATH, prefix=f'LP_{LP_cut_off:0.2f}_magnitude_images_{subject}')

# %% APPLY LOW_PASS FILTER TO ALL THE IMAGES AND SAVE AS A NEW DATASET 
import importlib
importlib.reload(dataset_utilities)

LP_cut_off = 0.5 # as ratio of frequences to keep
SAVE_PATH_LP_DATASET = os.path.join(os.path.dirname(pathlib.Path(DATASET_PATH)), f'LOW_PASS_FILTER_{LP_cut_off}_{os.path.basename(pathlib.Path(DATASET_PATH))}_{datetime.now().strftime("%y%m%d")}')
pathlib.Path(SAVE_PATH_LP_DATASET).mkdir(parents=True, exist_ok=True)

for set_name in ['training', 'validation', 'test']:
    # get files for this set 
    set_file_paths = list(dataset_split_df.loc[
                    dataset_split_df[f"fold_1"] == set_name
                ]["file_path"])
    set_files_labels = list(dataset_split_df.loc[
                    dataset_split_df[f"fold_1"] == set_name
                ]["target"])
    
    aus_dataset = dataset_utilities.PNGDatasetFromFolder(item_list=set_file_paths, labels=set_files_labels, return_file_path=True, transform=None)

    aus_dataset = iter(aus_dataset)
    for i in range(len(set_file_paths)):
        # print status
        print(f'Working on {set_name} set: file {i+1}/{len(set_file_paths)}         \r', end='')
        # get image
        img, label, _, file_path = next(aus_dataset)

        # process the image
        img = img.numpy().transpose(1,2,0)[:,:,0]

        # from https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        raw_magnitude = 20*np.log(np.abs(fshift))

        # build gaussian filter and apply
        rows, cols = img.shape
        crow,ccol = int(rows/2) , int(cols/2)
        gfilter = gkern(std=rows*LP_cut_off)

        fshift = fshift * gfilter
    
        f_ishift = np.fft.ifftshift(fshift)
        LP_img = np.fft.ifft2(f_ishift)
        LP_img = np.abs(LP_img)

        # save image
        LP_img = LP_img.astype(np.uint8)
        LP_img = Image.fromarray(LP_img)
        LP_img.save(os.path.join(SAVE_PATH_LP_DATASET, os.path.basename(file_path)))


# %% APPLY HIGH_PASS FILTER TO ALL THE IMAGES AND SAVE AS A NEW DATASET 
import importlib
importlib.reload(dataset_utilities)

LP_cut_off = 0.001 # as ratio of frequences to keep
SAVE_PATH_LP_DATASET = os.path.join(os.path.dirname(pathlib.Path(DATASET_PATH)), f'HIGH_PASS_FILTER_{LP_cut_off}_{os.path.basename(pathlib.Path(DATASET_PATH))}_{datetime.now().strftime("%y%m%d")}')
pathlib.Path(SAVE_PATH_LP_DATASET).mkdir(parents=True, exist_ok=True)

for set_name in ['training', 'validation', 'test']:
    # get files for this set 
    set_file_paths = list(dataset_split_df.loc[
                    dataset_split_df[f"fold_1"] == set_name
                ]["file_path"])
    set_files_labels = list(dataset_split_df.loc[
                    dataset_split_df[f"fold_1"] == set_name
                ]["target"])
    
    aus_dataset = dataset_utilities.PNGDatasetFromFolder(item_list=set_file_paths, labels=set_files_labels, return_file_path=True, transform=None)

    aus_dataset = iter(aus_dataset)
    for i in range(len(set_file_paths)):
        # print status
        print(f'Working on {set_name} set: file {i+1}/{len(set_file_paths)}         \r', end='')
        # get image
        img, label, _, file_path = next(aus_dataset)

        # process the image
        img = img.numpy().transpose(1,2,0)[:,:,0]

        # from https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        raw_magnitude = 20*np.log(np.abs(fshift))

        # build gaussian filter and apply
        rows, cols = img.shape
        crow,ccol = int(rows/2) , int(cols/2)
        gfilter = gkern(std=rows*LP_cut_off)
        # make high-pass filter
        gfilter_ones = np.ones_like(gfilter)
        gfilter_ones = gfilter_ones - gfilter

        fshift = fshift * gfilter_ones
    
        f_ishift = np.fft.ifftshift(fshift)
        LP_img = np.fft.ifft2(f_ishift)
        LP_img = np.abs(LP_img)

        # save image
        LP_img = LP_img.astype(np.uint8)
        LP_img = Image.fromarray(LP_img)
        LP_img.save(os.path.join(SAVE_PATH_LP_DATASET, os.path.basename(file_path)))

    
# %%
