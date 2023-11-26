# %%
# %% TEST SimCLR clustering

import os
import matplotlib.pyplot as plt
import pandas as pd
import torch 
import torchvision.transforms as T
import numpy as np
import cv2
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import importlib

import dataset_utilities
import model_bucket_CBTN_v1
from copy import deepcopy
import pathlib

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

import glob
import model_bucket_CBTN_v1


pl.seed_everything(42)
# %% LOOP ON MANY FOLDS

# pretrained_models_to_evaluate = ['ResNet50', 'ResNet18', 'ResNet9', 'ViT_16_b', 'ViT_32_b']
pretrained_models_to_evaluate = ['ResNet18']
set_to_plot = 'training'

for idy, model_version in enumerate(pretrained_models_to_evaluate):
    print(f'Model {idy+1} of {len(pretrained_models_to_evaluate)}')
    # %% DEFINE PATHS AND SETTINGS
    MODEL_INFO = {
        'pretraining_type': "classification", 
        'path': None,
    }

    DATASET_INFO = {
        'dataset_split_path' : '/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/trained_model_archive/TESTs_20231124/ResNet50_pretrained_True_frozen_True_0.5_LR_1e-05_BATCH_32_AUGMENTATION_True_OPTIM_adam_SCHEDULER_exponential_MLPNODES_0_t100754/REPETITION_1/data_split_information.csv',
        'set': set_to_plot,
        'nbr_samples_to_plot': None
    }

    SAVE_PATH = os.path.join(os.getcwd(), 'visuals')

    # build prefix sof saving
    SAVE_PREFIX = f'{model_version}_{set_to_plot}_set_ImageNet_pretraining_feature_plot_2'
    print(SAVE_PREFIX)
    # % BUILD DATASET
    # load dataset split file
    dataset_split = pd.read_csv(DATASET_INFO['dataset_split_path'])
    try:
        dataset_split = dataset_split.drop(columns=["level_0", "index"])
    except:
        print()

    # get mapping of the classes to one hot encoding
    unique_targe_classes = dict.fromkeys(pd.unique(dataset_split["target"]))
    one_hot_encodings = torch.nn.functional.one_hot(
        torch.tensor(list(range(len(unique_targe_classes)))))
    # build mapping between class and one hot encoding
    target_class_to_one_hot_mapping = dict(zip(unique_targe_classes, one_hot_encodings))

    files_for_inference = dataset_split.loc[
                    dataset_split[f"fold_{1}"]
                    == DATASET_INFO['set']
                ].reset_index()
                # make torch tensor labels
    labels = [
        target_class_to_one_hot_mapping[c]
        for c in list(files_for_inference["target"])
    ]
    labels_for_df = [list(l.numpy()) for l in labels]
    # add label to the dataframe
    files_for_inference.insert(
        files_for_inference.shape[1], "one_hot_encodig", labels_for_df
    )

    importlib.reload(dataset_utilities)
    img_mean = np.array([0.4451, 0.4262, 0.3959])
    img_std = np.array([0.2411, 0.2403, 0.2466])

    img_mean = np.array([0.5]*3)
    img_std = np.array([0.5]*3)

    preprocess = T.Compose(
            [
                T.Resize(size=(224, 224), antialias=True),
                T.ToTensor(),
                T.Normalize(
                    mean=img_mean,
                    std=img_std,
                ),
            ],
        )
    dataset = DataLoader(
                    dataset_utilities.PNGDatasetFromFolder(
                        list(files_for_inference["file_path"]),
                        transform=preprocess,
                        labels=labels,
                        return_file_path=False,
                    ),
                    num_workers=10,
                    batch_size=32,
                )

    # %% LOAD MODEL
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    if model_version.lower() == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT)
    elif model_version.lower() == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT)
    elif model_version.lower() == "resnet9":
        model = model_bucket_CBTN_v1.ResNet9()
        # laod model from .pth
        model.load_state_dict(torch.load(os.path.join(os.getcwd(), "pretrained_models/resnet9.pth")))
    elif model_version.lower() == 'vit_16_b':
        model = models.vit_b_16(weights='DEFAULT')
    elif model_version.lower() == 'vit_32_b':
        model = models.vit_b_32(weights='DEFAULT')


    # %% remove classification head
    if 'vit' not in model_version.lower():
        if model_version.lower() == 'resnet9':
            model.classifier[2] = torch.nn.Identity()
            model.classifier[3] = torch.nn.Identity()
        else:
            model.fc = torch.nn.Identity() 
    elif 'vit' in model_version.lower():
        model.heads = torch.nn.Identity()
    model.to(device)
    model.eval()

    # %% EMBED IMAGES
    @torch.no_grad()
    def prepare_data_features(model, data_loader, nbr_images_to_generate: int = None, device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")):
        # # Encode as many images as requested
        feats, labels = [], []
        nbr_images_to_generate = (
            nbr_images_to_generate if nbr_images_to_generate else len(data_loader.dataset)
        )
        counter = 0
        while counter < nbr_images_to_generate:
            for idx, (batch_imgs, batch_labels) in enumerate(data_loader):
                batch_imgs = batch_imgs.to(device)
                batch_feats = model(batch_imgs)
                feats.extend(batch_feats.detach().cpu().numpy())
                labels.extend(batch_labels.numpy())

                # update counter
                counter += batch_labels.shape[0]

                print(f"Processing {counter}\{nbr_images_to_generate}\r", end="")

                if counter >= nbr_images_to_generate:
                    break
        print("\n")
        
        return np.stack(feats), np.stack(labels)

    feats, labels = prepare_data_features(model, dataset, nbr_images_to_generate=None)
    str_labels = []
    for l in labels:
        for k,v in target_class_to_one_hot_mapping.items():
            if all(v == torch.tensor(l)):
                str_labels.append(k)

    # %% PLOT
    def plot_embeddings(emb,hue_labels, style_labels=None, tool='tsne', draw:bool=True, save_figure:str=None, save_path:str=None, prefix:str='Embeddings_cluster'):
        if tool=='tsne':
            tl=TSNE(n_components=3, perplexity=int(emb.shape[0]/6))
        else:
            tl=PCA(n_components=3)
        embedding=tl.fit_transform(emb)

        # create axis
        fig , axis = plt.subplots(nrows=2, ncols=2, figsize = (10, 10))


        # populate axis
        for idx, (ax, dim_indexes, view_names) in enumerate(zip(
            fig.axes, ([0,1], [0,2], [1,2]), ('dim_1 - vs - dim_2', 'dim_1 - vs - dim_3', 'dim_2 - vs - dim_3')
        )):
            sns.scatterplot(x= embedding[:,dim_indexes[0]], y = embedding[:,dim_indexes[1]], hue=hue_labels, style=style_labels, legend=False if idx !=2 else True, ax=ax)
            # set title
            ax.set_title(f'{tool.upper()} ({view_names})')

            # remove legend for all apart from last plot
            if idx == 2:
                ax.legend(loc='center left',ncol=3, bbox_to_anchor=(1.1, 0.5))
                plt.setp(ax.get_legend().get_texts(), fontsize='6')
        
        # hide last axis
        axis[1,1].axis('off')
        
        if save_figure:
            fig.savefig(os.path.join(save_path, f'{prefix}_{tool.upper()}.pdf'), dpi=100, bbox_inches='tight')
        if draw: 
            plt.show()
        else:
            plt.close(fig)

    labels = list(files_for_inference["target"])[0:feats.shape[0]]
    subject_ids = list(files_for_inference["subject_IDs"])[0:feats.shape[0]]

    plot_embeddings(feats,hue_labels=labels,style_labels=subject_ids, tool='tsne', draw=False, save_figure=True, save_path=SAVE_PATH, prefix=SAVE_PREFIX)
    plot_embeddings(feats,hue_labels=labels,style_labels=subject_ids, tool='pca', draw=False, save_figure=True, save_path=SAVE_PATH, prefix=SAVE_PREFIX)

# %%