import math
import os
import sys
import numpy as np
from typing import Any, Tuple, Union
from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch_optimizer as optim

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchmetrics
from torchvision import models

from torchsummary import summary
import monai
import evaluation_utilities


from scipy.special import softmax


# %% some utilities
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# %% GENERAL RESNET MODEL
def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def make_classification_mpl(
    in_features: int,
    mpl_nodes: list,
    dropout_rate: Union[list, float],
    nbr_classes: int,
    add_pooling_layer: bool = False,
):
    # utility that builds the classification layers based on the specifications
    classification_layers = torch.nn.Sequential()
    if add_pooling_layer:
        # add GlobalAveragePooling and flatten layer (ResNet9 needs it)
        classification_layers.append(torch.nn.AdaptiveMaxPool2d((1, 1)))
        classification_layers.append(torch.nn.Flatten())

    # add layers
    for nbr_nodes, dropout_rate in zip(mpl_nodes, dropout_rate):
        classification_layers.append(
            torch.nn.Linear(in_features=in_features, out_features=nbr_nodes)
        )
        classification_layers.append(nn.ReLU())
        classification_layers.append(torch.nn.Dropout(p=dropout_rate, inplace=False))
        # update in_features
        in_features = nbr_nodes

    # add last layer
    classification_layers.append(torch.nn.Linear(in_features, nbr_classes))

    return classification_layers


class ResNet9(nn.Module):
    def __init__(self, nbr_classes: int = 10, in_channels: int = 3):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, nbr_classes),
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


class ResNets(torch.nn.Module):
    def __init__(
        self,
        nbr_classes: int,
        version: str = "resnet50",
        pretrained: bool = True,
        freeze_percentage: float = 1.0,
        mpl_nodes: Union[list, tuple] = [1024],
        dropout_rate: float = 0.6,
    ):
        super(ResNets, self).__init__()

        self.mpl_nodes = mpl_nodes
        if isinstance(dropout_rate, float):
            self.dropout_rate = [dropout_rate for i in range(len(mpl_nodes))]
        else:
            assert len(dropout_rate) == len(
                mpl_nodes
            ), f"Single modality classification model build: the number of dropout rates and the number of mpl layers does not match. Given {len(dropout_rate)} and {len(nodes_in_mpl)}"
            self.dropout_rate = dropout_rate
        self.nbr_classes = nbr_classes
        # load model
        if version == "resnet50":
            self.model = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
        elif version == "resnet18":
            self.model = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
        elif version == "resnet9":
            self.model = ResNet9()
            if pretrained:
                # laod model from .pth
                self.model.load_state_dict(
                    torch.load(
                        os.path.join(os.getcwd(), "pretrained_models/resnet9.pth")
                    )
                )

        # ## freeze parts of the model as specified
        if pretrained:
            model_children_to_freeze = {
                "resnet50": {
                    1.0: {
                        "children_to_freeze": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                        "string_to_print": "Freezing all the model (children 0 to 8).",
                    },
                    0.8: {
                        "children_to_freeze": [0, 1, 2, 3, 4, 5, 6],
                        "string_to_print": "Freezing children 0 to 6 (7 is training).",
                    },
                    0.4: {
                        "children_to_freeze": [0, 1, 2, 3, 4, 5],
                        "string_to_print": "Freezing children 0 to 5 (7, 6 are training)",
                    },
                    0.20: {
                        "children_to_freeze": [0, 1, 2, 3, 4],
                        "string_to_print": "Freezing children 0 to 4 (7, 6, 5 are training)",
                    },
                    0.05: {
                        "children_to_freeze": [0, 1, 2, 3],
                        "string_to_print": "Freezing children 0 to 3 (7, 6, 5, 4 are training)",
                    },
                    0.00: {
                        "children_to_freeze": [],
                        "string_to_print": "All the model is training.",
                    },
                },
                "resnet18": {
                    1.0: {
                        "children_to_freeze": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                        "string_to_print": "Freezing all the model (children 0 to 8).",
                    },
                    0.8: {
                        "children_to_freeze": [0, 1, 2, 3, 4, 5, 6],
                        "string_to_print": "Freezing children 0 to 6 (7 is training).",
                    },
                    0.6: {
                        "children_to_freeze": [0, 1, 2, 3, 4, 5],
                        "string_to_print": "Freezing children 0 to 5 (7, 6 are training)",
                    },
                    0.4: {
                        "children_to_freeze": [0, 1, 2, 3, 4],
                        "string_to_print": "Freezing children 0 to 4 (7, 6, 5 are training)",
                    },
                    0.05: {
                        "children_to_freeze": [0, 1, 2, 3],
                        "string_to_print": "Freezing children 0 to 3 (7, 6, 5, 4 are training)",
                    },
                    0.00: {
                        "children_to_freeze": [],
                        "string_to_print": "All the model is training.",
                    },
                },
                "resnet9": {
                    1.0: {
                        "children_to_freeze": [0, 1, 2, 3, 4, 5],
                        "string_to_print": "Freezing all the model (children 0 to 6).",
                    },
                    0.8: {
                        "children_to_freeze": [0, 1, 2, 3, 4],
                        "string_to_print": "Freezing children 0 to 4 (5 is training).",
                    },
                    0.6: {
                        "children_to_freeze": [0, 1, 2, 3],
                        "string_to_print": "Freezing children 0 to 3 (5, 4 are training)",
                    },
                    0.4: {
                        "children_to_freeze": [0, 1, 2],
                        "string_to_print": "Freezing children 0 to 2 (5, 4, 3 are training)",
                    },
                    0.2: {
                        "children_to_freeze": [0, 1],
                        "string_to_print": "Freezing children 0 to 1 (5, 4, 3, 2 are training)",
                    },
                    0.05: {
                        "children_to_freeze": [0],
                        "string_to_print": "Freezing children 0 (5, 4, 3, 2, 1 are training)",
                    },
                    0.00: {
                        "children_to_freeze": [],
                        "string_to_print": "All the model is training.",
                    },
                },
            }

            # fix the freeze_percentage to be the closest to the allowed ones
            freeze_percentage = find_nearest(
                list(model_children_to_freeze[version].keys()), freeze_percentage
            )

            # freeze model using freeze_percentage
            child_idx_for_summary = []
            for idx, child in enumerate(self.model.children()):
                # freeze if this children is in the list of the children to freez for this percentage
                if (
                    idx
                    in model_children_to_freeze[version][freeze_percentage][
                        "children_to_freeze"
                    ]
                ):
                    for param in child.parameters():
                        param.requires_grad = False
                    child_idx_for_summary.append(idx)

            # print status (here print both the automatic and the default string to check if all is good)
            print(
                f'({freeze_percentage}) {model_children_to_freeze[version][freeze_percentage]["string_to_print"]}'
            )
            print(
                f"({freeze_percentage}) (Automatic print): Freezing children {child_idx_for_summary}"
            )

        # build classifier
        if version == "resnet9":
            self.model.classifier = make_classification_mpl(
                in_features=512,
                mpl_nodes=self.mpl_nodes,
                dropout_rate=self.dropout_rate,
                nbr_classes=self.nbr_classes,
                add_pooling_layer=True,
            )
        else:
            self.model.fc = make_classification_mpl(
                in_features=self.model.fc.in_features,
                mpl_nodes=self.mpl_nodes,
                dropout_rate=self.dropout_rate,
                nbr_classes=self.nbr_classes,
            )

    def forward(self, x):
        return self.model(x)


# %% INCEPTION
class InceptionV3(torch.nn.Module):
    def __init__(
        self,
        nbr_classes: int,
        pretrained: bool = True,
        freeze_percentage: float = 1.0,
        mpl_nodes: Union[list, tuple] = [1024],
        dropout_rate: float = 0.6,
    ):
        super(InceptionV3, self).__init__()

        self.mpl_nodes = mpl_nodes
        if isinstance(dropout_rate, float):
            self.dropout_rate = [dropout_rate for i in range(len(mpl_nodes))]
        else:
            assert len(dropout_rate) == len(
                mpl_nodes
            ), f"Single modality classification model build: the number of dropout rates and the number of mpl layers does not match. Given {len(dropout_rate)} and {len(nodes_in_mpl)}"
            self.dropout_rate = dropout_rate
        self.nbr_classes = nbr_classes
        # load model
        self.model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT if pretrained else None
        )
        self.model.aux_logits = False

        # ## freeze parts of the model as specified
        if pretrained:
            # freeze model using freeze_percentage
            for idx, child in enumerate(self.model.children()):
                # freeze if this children is in the list of the children to freez for this percentage
                for param in child.parameters():
                    param.requires_grad = False

        # build classifier
        self.model.fc = make_classification_mpl(
            in_features=self.model.fc.in_features,
            mpl_nodes=self.mpl_nodes,
            dropout_rate=self.dropout_rate,
            nbr_classes=self.nbr_classes,
        )

    def _make_classification_mpl(
        in_features: int,
        mpl_nodes: list,
        dropout_rate: Union[list, float],
        nbr_classes: int,
        add_pooling_layer: bool = False,
    ):
        # utility that builds the classification layers based on the specifications
        classification_layers = torch.nn.Sequential()
        if add_pooling_layer:
            # add GlobalAveragePooling and flatten layer (ResNet9 needs it)
            classification_layers.append(torch.nn.AdaptiveMaxPool2d((1, 1)))
            classification_layers.append(torch.nn.Flatten())

        # add layers
        for nbr_nodes, dropout_rate in zip(mpl_nodes, dropout_rate):
            classification_layers.append(
                torch.nn.Linear(in_features=in_features, out_features=nbr_nodes)
            )
            classification_layers.append(nn.ReLU())
            classification_layers.append(
                torch.nn.Dropout(p=dropout_rate, inplace=False)
            )
            # update in_features
            in_features = nbr_nodes

        # add last layer
        classification_layers.append(torch.nn.Linear(in_features, nbr_classes))

        return classification_layers

    def forward(self, x):
        return self.model(x)


# %% CUSTOM MODEL
class CustomModel(torch.nn.Module):
    def __init__(
        self,
        nbr_classes: int,
        conv_dropout_rate: float = 0.4,
        dense_dropout_rate: float = 0.2,
    ):
        super().__init__()

        # build model as described in https://github.com/IulianEmilTampu/qMRI_and_DL/blob/main/Tumor_detection_scripts/training_scripts/detection_models.py

        self.nbr_classes = nbr_classes
        self.filters_in_encoder = [64, 128, 256, 512]
        self.dense_dropout_rate = dense_dropout_rate

        if isinstance(conv_dropout_rate, float):
            self.conv_dropout_rate = [
                conv_dropout_rate for i in range(len(self.filters_in_encoder))
            ]
        else:
            assert len(conv_dropout_rate) == len(
                self.filters_in_encoder
            ), f"2D-SDM4 model build: the number of dropout rates and the number of encoder layers does not match. Given {len(dropout_rate)} and {len(self.filters_in_encoder)}"
            self.conv_dropout_rate = conv_dropout_rate

        # build encoder
        self.encoder = self._buind_encoder()
        # build classifier
        self.classification_head = self._buind_classifier_head(nbr_nodes=256)

    def _buind_encoder(self):
        # utility that builds the encoder and classification layers
        encoder_layers = torch.nn.ModuleList()
        in_features = 3  # this are the number of channels in the input image (RGB)

        # add layers
        for filters, dropout_rate in zip(
            self.filters_in_encoder, self.conv_dropout_rate
        ):
            encoder_layer = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=in_features,
                    out_channels=filters,
                    kernel_size=3,
                    padding="same",
                ),
                torch.nn.InstanceNorm2d(num_features=filters),
                torch.nn.ReLU(inplace=False),
                torch.nn.Dropout(p=dropout_rate, inplace=False),
                torch.nn.Conv2d(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=3,
                    padding="same",
                ),
                torch.nn.InstanceNorm2d(num_features=filters),
                torch.nn.ReLU(inplace=False),
                torch.nn.Dropout(p=dropout_rate, inplace=False),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
            )

            # add to the classification layers
            encoder_layers.append(encoder_layer)
            # update in_features
            in_features = filters

        return encoder_layers

    def _buind_classifier_head(self, nbr_nodes: int = 512):
        # utility that builds the encoder and classification layers
        classification_head = torch.nn.ModuleList()
        classification_head.append(
            torch.nn.BatchNorm1d(num_features=self.filters_in_encoder[-1])
        )
        classification_head.append(
            torch.nn.Linear(self.filters_in_encoder[-1], nbr_nodes)
        )
        classification_head.append(
            torch.nn.Dropout(p=self.dense_dropout_rate, inplace=False)
        )
        classification_head.append(torch.nn.Linear(nbr_nodes, self.nbr_classes))

        return classification_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input through the encoder
        for enc_layer in self.encoder:
            x = enc_layer(x)

        # flatten using GlobalAveragePooling (mean along the feature map dimensions)
        x = torch.mean(x, dim=[2, 3])

        # through the classification head
        for classification_layer in self.classification_head:
            x = classification_layer(x)
        return x


# %% RAD RESNET MODEL
# from https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/losses/perceptual.py#L361
class RadImageNet(torch.nn.Module):
    """
    Component to perform the perceptual evaluation with the networks pretrained on RadImagenet (pretrained by Mei, et
    al. "RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning"). This class
    uses torch Hub to download the networks from "Warvito/radimagenet-models".

    Args:
        net: {``"radimagenet_resnet50"``}
            Specifies the network architecture to use. Defaults to ``"radimagenet_resnet50"``.
        verbose: if false, mute messages from torch Hub load function.
    """

    def __init__(
        self,
        nbr_classes: int,
        net: str = "radimagenet_resnet50",
        freeze_percentage=1.0,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        # load RadResNet
        self.model = torch.hub.load(
            "Warvito/radimagenet-models", model=net, verbose=verbose
        )

        # freeze model if needed
        model_children_to_freeze = {
            "resnet50": {
                1.0: {
                    "children_to_freeze": [0, 1, 2, 3, 4, 5, 6, 7],
                    "string_to_print": "Freezing all the model (children 0 to 7).",
                },
                0.8: {
                    "children_to_freeze": [0, 1, 2, 3, 4, 5, 6],
                    "string_to_print": "Freezing children 0 to 6 (7 is training).",
                },
                0.4: {
                    "children_to_freeze": [0, 1, 2, 3, 4, 5],
                    "string_to_print": "Freezing children 0 to 5 (7, 6 are training)",
                },
                0.20: {
                    "children_to_freeze": [0, 1, 2, 3, 4],
                    "string_to_print": "Freezing children 0 to 4 (7, 6, 5 are training)",
                },
                0.05: {
                    "children_to_freeze": [0, 1, 2, 3],
                    "string_to_print": "Freezing children 0 to 3 (7, 6, 5, 4 are training)",
                },
                0.00: {
                    "children_to_freeze": [],
                    "string_to_print": "All the model is training.",
                },
            }
        }

        # fix the freeze_percentage to be the closest to the allowed ones
        freeze_percentage = find_nearest(
            list(model_children_to_freeze["resnet50"].keys()), freeze_percentage
        )

        # freeze model using freeze_percentage
        child_idx_for_summary = []
        for idx, child in enumerate(self.model.children()):
            # freeze if this children is in the list of the children to freez for this percentage
            if (
                idx
                in model_children_to_freeze["resnet50"][freeze_percentage][
                    "children_to_freeze"
                ]
            ):
                for param in child.parameters():
                    param.requires_grad = False
                child_idx_for_summary.append(idx)

        # print status (here print both the automatic and the default string to check if all is good)
        print(
            f'({freeze_percentage}) {model_children_to_freeze["resnet50"][freeze_percentage]["string_to_print"]}'
        )
        print(
            f"({freeze_percentage}) (Automatic print): Freezing children {child_idx_for_summary}"
        )

        self.nbr_classes = nbr_classes

        # build classifier
        self.classification_head = self._buind_classifier_head()

    def _buind_classifier_head(self):
        # utility that builds the encoder and classification layers
        classification_head = torch.nn.ModuleList()
        classification_head.append(torch.nn.Dropout(p=0.2, inplace=False))
        classification_head.append(torch.nn.Linear(2048, 1024))
        classification_head.append(torch.nn.BatchNorm1d(num_features=1024))
        classification_head.append(torch.nn.ReLU(inplace=False))
        classification_head.append(torch.nn.Linear(1024, self.nbr_classes))

        return classification_head

    def subtract_mean(self, x: torch.Tensor) -> torch.Tensor:
        mean = [0.406, 0.456, 0.485]
        x[:, 0, :, :] -= mean[0]
        x[:, 1, :, :] -= mean[1]
        x[:, 2, :, :] -= mean[2]
        return x

    def normalize_tensor(self, x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
        return x / (norm_factor + eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        We expect that the input is normalised between [0, 1]. Given the preprocessing performed during the training at
        https://github.com/BMEII-AI/RadImageNet, we make sure that the input and target have 3 channels, reorder it from
         'RGB' to 'BGR', and then remove the mean components of each input data channel. The outputs are normalised
        across the channels, and we obtain the mean from the spatial dimensions (similar approach to the lpips package).
        """

        # # Subtract mean used during training
        x = self.subtract_mean(x)

        # Get model outputs
        x = self.model(x)

        # # # Normalise through the channels
        x = self.normalize_tensor(x)

        # flatten using GlobalAveragePooling (mean along the feature map dimensions)
        x = torch.mean(x, dim=[2, 3])

        # thtourh the classification head
        for classification_layer in self.classification_head:
            x = classification_layer(x)
        return x


# %% ViT
class ViT_nomai(torch.nn.Module):
    def __init__(
        self,
        nbr_classes: int,
        spatial_dims: int = 2,
        in_channels: int = 3,
        img_size: Union[list, tuple] = (224, 224),
        proj_type: str = "conv",
        pos_embed_type: str = "sincos",
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.model = monai.networks.nets.ViT(
            spatial_dims=spatial_dims,
            num_classes=nbr_classes,
            patch_size=patch_size,
            in_channels=in_channels,
            img_size=img_size,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            classification=True,
        )

    def forward(self, x):
        return self.model(x)[0]


class ViTs(torch.nn.Module):
    def __init__(
        self,
        nbr_classes: int,
        version: str = "vit_b_16",
        pretrained: bool = True,
        freeze_percentage: float = 1.0,
    ) -> None:
        super().__init__()

        # load the right ViT version
        if version == "vit_b_16":
            self.model = models.vit_b_16(weights="DEFAULT" if pretrained else None)
        elif version == "vit_b_32":
            self.model = models.vit_b_32(weights="DEFAULT" if pretrained else None)
        else:
            raise ValueError(
                f"The given ViT version is not among the one supported. Add the version here if needed. Given {version}"
            )

        if pretrained:
            # get the number of encoder layers to freeze
            nbr_enc_layer_to_freeze = int(
                len(self.model.encoder.layers) * freeze_percentage
            )
            # freeze model
            for nbr_enc_layer in range(nbr_enc_layer_to_freeze):
                for param in self.model.encoder.layers[nbr_enc_layer].parameters():
                    param.requires_grad = False
            # print status
            print(
                f"({freeze_percentage}) (Automatic print): Freezing {nbr_enc_layer_to_freeze} encoding blocks out of {len(self.model.encoder.layers)}"
            )

        # adapt the classification head to the nunber of classification tasks
        self.model.heads[0] = torch.nn.Linear(in_features=768, out_features=nbr_classes)

    def forward(self, x):
        return self.model(x)


# %% BUILD CLASSIFIER FROM SimCLR model


class ClassifierFromSimCLR(torch.nn.Module):
    def __init__(
        self,
        SimCLR_model_path,
        nbr_classes,
        freeze_percentage: float = 1.0,
        mpl_nodes: Union[list, tuple] = (1024),
        dropout_rate: Union[list, tuple, float] = 0.2,
    ):
        super().__init__()

        self.SimCLR_model_path = SimCLR_model_path
        self.nbr_classes = nbr_classes
        self.freeze_percentage = freeze_percentage
        self.mpl_nodes = mpl_nodes
        self.dropout_rate = dropout_rate

        # laod model
        if not os.path.isfile(self.SimCLR_model_path):
            raise ValueError(
                f"The given model path for the SimCLR model is not available. Please check. Given {self.SimCLR_model_path}"
            )

        self.model = torch.load(self.SimCLR_model_path)
        self.model = self.model.convnet

        # build classifier head
        if isinstance(self.dropout_rate, float):
            self.dropout_rate = [self.dropout_rate for i in range(len(self.mpl_nodes))]
        else:
            assert len(self.dropout_rate) == len(
                mpl_nodes
            ), f"Single modality classification model build: the number of dropout rates and the number of mpl layers does not match. Given {len(dropout_rate)} and {len(nodes_in_mpl)}"
            self.dropout_rate = self.dropout_rate

        self.model.fc = make_classification_mpl(
            in_features=self.model.fc[0][0].in_features,
            mpl_nodes=self.mpl_nodes,
            dropout_rate=self.dropout_rate,
            nbr_classes=self.nbr_classes,
        )

    def forward(self, x):
        return self.model(x)


# %% GENERAL LiT WRAPPER FOR CLASSIFICATION
class LitModelWrapper(pl.LightningModule):
    def __init__(
        self,
        version: str = "resnet50",
        pretrained: bool = True,
        in_channels: int = 3,
        nbr_classes: int = 3,
        class_weights: Union[list, tuple] = None,
        learning_rate: float = 3e-4,
        freeze_percentage: float = 1.0,
        image_mean: Union[list, tuple] = (0.4451, 0.4262, 0.3959),
        image_std: Union[list, tuple] = (0.2411, 0.2403, 0.2466),
        optimizer: str = "sgd",  # adamw, adam, sgd
        scheduler: str = "statis",  # linear, cyclical, reduce_on_plateau, exponential, static
        use_look_ahead_wrapper: bool = False,
        use_regularization: bool = False,
        mpl_nodes: Union[list, tuple] = (1024),
        use_SimCLR_pretrained_model: bool = False,
        SimCLR_model_path: str = None,
    ):
        super(LitModelWrapper, self).__init__()

        self.in_channels = in_channels
        self.nbr_classes = nbr_classes
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_look_ahead_wrapper = use_look_ahead_wrapper
        self.use_regularization = use_regularization
        self.version = version
        self.mpl_nodes = mpl_nodes

        # define model based on the version
        if use_SimCLR_pretrained_model:
            # build model based on SimCLR pretrained model
            self.model = ClassifierFromSimCLR(
                SimCLR_model_path=SimCLR_model_path,
                nbr_classes=self.nbr_classes,
                freeze_percentage=freeze_percentage,
                mpl_nodes=self.mpl_nodes,
                dropout_rate=0.2,
            )
        else:
            if any(
                [version == "resnet50", version == "resnet18", version == "resnet9"]
            ):
                self.model = ResNets(
                    nbr_classes=nbr_classes,
                    version=version,
                    pretrained=pretrained,
                    freeze_percentage=freeze_percentage,
                    mpl_nodes=mpl_nodes,
                )
            elif version == "2dsdm4":
                self.model = CustomModel(
                    nbr_classes=nbr_classes,
                    conv_dropout_rate=0.4,
                    dense_dropout_rate=0.2,
                )
            elif version == "radresnet":
                self.model = RadImageNet(
                    nbr_classes=nbr_classes, freeze_percentage=freeze_percentage
                )
            elif version == "inceptionv3":
                self.model = InceptionV3(
                    nbr_classes=nbr_classes,
                    pretrained=pretrained,
                    freeze_percentage=freeze_percentage,
                    mpl_nodes=mpl_nodes,
                )
            elif any([version == "vit_b_16", version == "vit_b_32"]):
                self.model = ViTs(
                    nbr_classes=nbr_classes,
                    version=version,
                    pretrained=pretrained,
                    freeze_percentage=freeze_percentage,
                )
            else:
                raise ValueError(
                    f"The given model version is not implemented (yet). Given {version}.\nYou can add your model implementation in the model_bucket_CBTN_v1.py file and call it in the LitModeWrapper."
                )

        # check class weights
        if class_weights is not None:
            if len(class_weights) != nbr_classes:
                raise ValueError(
                    f"Class weights and number of classes do not match. Given {class_weights} (class weights) and {nbr_classes} (number of classes)"
                )
            else:
                self.class_weights = torch.Tensor(class_weights)
        else:
            self.class_weights = torch.Tensor([1 for i in range(nbr_classes)])

        # define loss
        if nbr_classes == 2:
            self.loss_fn = nn.BCEWithLogitsLoss(weight=self.class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

        self.save_hyperparameters(
            logger=True,
        )

        # what to log during training, validation and testing
        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(
                    task="multiclass",
                    num_classes=nbr_classes,
                ),
                torchmetrics.MatthewsCorrCoef(
                    task="multiclass",
                    num_classes=nbr_classes,
                ),
            ]
        )

        self.train_metrics = metrics.clone(prefix="ptl/train_")
        self.valid_metrics = metrics.clone(prefix="ptl/valid_")
        self.test_metrics = metrics.clone(prefix="ptl/test_")

        # for the confusion matrix plot
        self.training_step_y_hats = []
        self.training_step_ys = []
        self.validation_step_y_hats = []
        self.validation_step_ys = []
        self.test_step_y_hats = []
        self.test_step_ys = []

        # for saving some training, validation, testing images
        self.validation_step_img = []
        self.training_step_img = []
        self.test_step_img = []

        # mean and std for rescaling image input (this is for visualization)
        self.image_mean = torch.Tensor(image_mean).unsqueeze(-1).unsqueeze(-1)
        self.image_std = torch.Tensor(image_std).unsqueeze(-1).unsqueeze(-1)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # set optimizer and scheduler based on settings
        if self.optimizer == "adamw":
            if all([self.use_regularization, self.version == "2dsdm4"]):
                optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr=self.learning_rate, weight_decay=0.001
                )
            else:
                optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr=self.learning_rate
                )
        elif self.optimizer == "adam":
            if all([self.use_regularization, self.version == "2dsdm4"]):
                optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=self.learning_rate, weight_decay=0.001
                )
            else:
                optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=self.learning_rate
                )
        else:
            if all([self.use_regularization, self.version == "2dsdm4"]):
                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.learning_rate,
                    weight_decay=0.001,
                    momentum=0.9,
                )
            else:
                optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=self.learning_rate, momentum=0.9
                )

        if self.use_look_ahead_wrapper:
            optimizer = optim.Lookahead(optimizer, k=5, alpha=0.5)

        # define scheduler
        if self.scheduler == "static":
            return optimizer
        else:
            if self.scheduler == "reduce_on_plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.1,
                    patience=10,
                )
            elif self.scheduler == "cyclical":
                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=self.learning_rate,
                    max_lr=self.learning_rate * 10,
                    # scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
                    cycle_momentum=False,
                )
            elif self.scheduler == "linear_decay":
                scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer)
            elif self.scheduler == "exponential":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=0.99,
                )

            return [optimizer], [scheduler]

    def log_gradients(self, step=None):
        for name, value in self.model.named_parameters():
            if "bn" not in name:
                if value.grad is not None:
                    self.loggers[0].experiment.add_histogram(
                        name + "/grad", value.grad.cpu(), step
                    )

    def log_weights(self, step=None):
        for name, value in self.model.named_parameters():
            if "bn" not in name:
                self.loggers[0].experiment.add_histogram(
                    name + "/weights", value.cpu(), step
                )

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)

        # compute metrics
        self.train_metrics.update(
            torch.argmax(torch.nn.functional.softmax(preds, dim=1), dim=1),
            torch.argmax(y, dim=1),
        )
        self.log(
            "ptl/train_classification_loss", loss.item(), on_epoch=True, sync_dist=True
        )
        # log gradients
        if self.trainer.global_step % 400 == 0:
            self.log_gradients(step=self.trainer.global_step)
            self.log_weights(step=self.trainer.global_step)

        # save one batch of images
        if len(self.training_step_img) == 0:
            self.training_step_img.append(x.to("cpu"))
        # save ground truth and predicstions for confusion matrix plot
        self.training_step_y_hats.append(preds)
        self.training_step_ys.append(y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)

        # compute metrics
        self.valid_metrics.update(
            torch.argmax(torch.nn.functional.softmax(preds, dim=1), dim=1),
            torch.argmax(y, dim=1),
        )
        self.log(
            "ptl/valid_classification_loss", loss.item(), on_epoch=True, sync_dist=True
        )

        # save images for tensorboard saving
        if len(self.validation_step_img) == 0:
            self.validation_step_img.append(x.to("cpu"))
        # save ground truth and predicstions for confusion matrix plot
        self.validation_step_y_hats.append(preds)
        self.validation_step_ys.append(y)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)

        # compute metrics
        self.test_metrics.update(
            torch.argmax(torch.nn.functional.softmax(preds, dim=1), dim=1),
            torch.argmax(y, dim=1),
        )
        self.log(
            "ptl/test_classification_loss", loss.item(), on_epoch=True, sync_dist=True
        )

        # save ground truth and predicstions for confusion matrix plot
        self.test_step_y_hats.append(preds)
        self.test_step_ys.append(y)

        # save images for tensorboard saving
        # save one batch of images
        if len(self.test_step_img) == 0:
            self.test_step_img.append(x.to("cpu"))

    def on_train_epoch_end(self) -> None:
        # save metrics
        output = self.train_metrics.compute()
        self.log_dict(output)

        # fix metrics for confusion matric title
        output = [f'{k}: {v.to("cpu").numpy():0.2f}\n' for k, v in output.items()]

        # save confusion matrix
        self._save_confusion_matrix(
            self.training_step_ys,
            self.training_step_y_hats,
            fig_name="Confusion matrix training",
            metrics=output,
        )

        # save images
        _imgs = torch.cat(self.training_step_img)
        self.save_imgs(
            _imgs,
            title="Example_train_batch",
        )

        # reset
        self.training_step_img = []
        self.train_metrics.reset()
        self.training_step_ys, self.training_step_y_hats = [], []

    def on_validation_epoch_end(self) -> None:
        # save metrics
        output = self.valid_metrics.compute()
        self.log_dict(output)

        # fix metrics for confusion matric title
        output = [f'{k}: {v.to("cpu").numpy():0.2f}\n' for k, v in output.items()]

        # save confusion matrix
        self._save_confusion_matrix(
            self.validation_step_ys,
            self.validation_step_y_hats,
            fig_name="Confusion matrix validation",
            metrics=output,
        )

        # save image batch
        _imgs = torch.cat(self.validation_step_img)
        self.save_imgs(
            _imgs,
            title="Example_validation_batch",
        )

        # reset
        self.validation_step_img = []
        self.valid_metrics.reset()
        self.validation_step_ys, self.validation_step_y_hats = [], []

    def on_test_epoch_end(self):
        # save metrics
        output = self.test_metrics.compute()
        self.log_dict(output)

        # fix metrics for confusion matric title
        output = [f'{k}: {v.to("cpu").numpy():0.2f}\n' for k, v in output.items()]

        # save confusion matrix
        self._save_confusion_matrix(
            self.test_step_ys,
            self.test_step_y_hats,
            fig_name="Confusion matrix test",
            metrics=output,
        )

        # save test images
        _imgs = torch.cat(self.test_step_img)
        self.save_imgs(
            _imgs,
            title="Example_test_batch",
        )

    def _save_confusion_matrix(
        self, y, y_hat, fig_name="Confusion_matrix", metrics=None
    ):
        # save confusion matrix
        y_hat = torch.cat(y_hat).cpu()
        y = torch.cat(y).cpu()

        confusion_matrix = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=self.nbr_classes, threshold=0.05
        )

        confusion_matrix(torch.argmax(y_hat, dim=1), torch.argmax(y, dim=1))

        confusion_matrix_computed = (
            confusion_matrix.compute().detach().cpu().numpy().astype(int)
        )

        df_cm = pd.DataFrame(confusion_matrix_computed)
        fig_, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, cmap="Blues", annot_kws={"size": 20}, ax=ax)
        ax.set_title(metrics)
        fig_ = fig_.get_figure()
        self.loggers[0].experiment.add_figure(fig_name, fig_, self.current_epoch)
        plt.close(fig_)

        # # save confusion matrix using evaluation utilities (this is for debugging)
        # evaluation_utilities.plotConfusionMatrix(
        #     GT=y.detach().numpy(),
        #     PRED=softmax(y_hat.detach().numpy()),
        #     classes=["ASTR", "EP", "MED"],
        #     savePath="/flush2/iulta54/Code/P5-PediatricBrainTumorClassification_CBTN_v1/trained_model_archive/TESTs_20231127",
        #     saveName=f"{fig_name}_{self.trainer.global_step}",
        #     draw=False,
        # )

    def save_imgs(self, imgs: torch.Tensor, title: str = "TB_saved_img"):
        with torch.no_grad():
            imgs = imgs.detach()
            # print(f"\nmean: {torch.mean(imgs)}, std: {torch.std(imgs)}")
            imgs = imgs * self.image_std + self.image_mean
            imgs = torchvision.utils.make_grid(imgs).cpu()
            self.loggers[0].experiment.add_image(
                title, imgs, self.trainer.current_epoch
            )


# %% LiT WRAPPER FOR SimCLR training


class SimCLRModelWrapper(pl.LightningModule):
    def __init__(
        self,
        version: str = "resnet50",
        pretrained: bool = False,
        freeze_percentage: float = 1.0,
        hidden_dim: int = 128,
        lr: float = 0.001,
        temperature: float = 0.07,
        weight_decay: float = 1e-4,
        max_epochs=500,
    ):
        super().__init__()
        self.save_hyperparameters()
        assert (
            self.hparams.temperature > 0.0
        ), "The temperature must be a positive float!"

        # Base model f(.)
        # define model based on the version
        if any([version == "resnet50", version == "resnet18", version == "resnet9"]):
            self.convnet = ResNets(
                nbr_classes=4 * hidden_dim,
                version=version,
                pretrained=pretrained,
                freeze_percentage=freeze_percentage,
                mpl_nodes=[],
            ).model
        elif version == "2dsdm4":
            self.convnet = CustomModel(
                nbr_classes=4 * hidden_dim,
                conv_dropout_rate=0.4,
                dense_dropout_rate=0.2,
            )
        elif version == "radresnet":
            self.convnet = RadImageNet(
                nbr_classes=nbr_classes, freeze_percentage=freeze_percentage
            ).model
        elif version == "inceptionv3":
            self.convnet = InceptionV3(
                nbr_classes=4 * hidden_dim,
                pretrained=pretrained,
                freeze_percentage=freeze_percentage,
                mpl_nodes=mpl_nodes,
            ).model
        elif any([version == "vit_b_16", version == "vit_b_32"]):
            self.convnet = ViTs(
                nbr_classes=4 * hidden_dim,
                version=version,
                pretrained=pretrained,
                freeze_percentage=freeze_percentage,
            )
        else:
            raise ValueError(
                f"The given model version is not implemented (yet). Given {version}.\nYou can add your model implementation in the model_bucket_CBTN_v1.py file and call it in the LitModeWrapper."
            )

        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        # for saving some training, validation, testing images
        self.validation_step_img = []
        self.training_step_img = []
        self.test_step_img = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.convnet(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log("ptl/" + mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [
                cos_sim[pos_mask][:, None],  # First position positive example
                cos_sim.masked_fill(pos_mask, -9e15),
            ],
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log("ptl/" + mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log("ptl/" + mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log("ptl/" + mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

        # # save one batch of images
        # if len(self.training_step_img) == 0:
        #     self.training_step_img.append(batch[0].to("cpu"))

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")

        # # save one batch of images
        # if len(self.validation_step_img) == 0:
        #     self.validation_step_img.append(x.to("cpu"))

    # TODO fix image plotting during training
    # def on_train_epoch_end(self) -> None:
    #     _imgs = torch.cat(self.training_step_img)
    #     self.save_imgs(
    #         _imgs,
    #         title="Example_train_batch",
    #     )
    #     return

    # def on_validation_epoch_end(self) -> None:
    #     return

    # def save_imgs(self, imgs_view_1: torch.Tensor, imgs_view_2: torch.Tensor, title: str = "TB_saved_img"):
    #     with torch.no_grad():
    #         imgs_view_1, imgs_view_2 = imgs_view_1.detach(), imgs_view_2.detach()
    #         imgs_view = imgs_view_1
    #         imgs_view.extend(imgs_view_2)
    #         imgs_view = torchvision.utils.make_grid(imgs_view, nrow=int(len(imgs_view)/2), normalize=True, pad_value=0.9).cpu()
    #         self.loggers[0].experiment.add_image(
    #             title, imgs_view, self.trainer.current_epoch
    #         )
