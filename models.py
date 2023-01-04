# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision


# class ConvBlock(nn.Module):
#     def __init__(
#         self,
#         input_channels: int,
#         output_channels: int,
#         kernel_size: int = 3,
#     ):
#         super().__init__()
#         normalization = nn.BatchNorm2d(num_features=output_channels)
#         # normalization = nn.LayerNorm(num_features=output_channels)
#         activation = nn.ReLU()
#         self.block = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=input_channels,
#                 out_channels=output_channels,
#                 kernel_size=kernel_size,
#             ),
#             normalization,
#             activation,
#             nn.Dropout2d(p=0.1),
#             nn.Conv2d(
#                 in_channels=output_channels,
#                 out_channels=output_channels,
#                 kernel_size=kernel_size,
#             ),
#             normalization,
#             activation,
#         )

#     def forward(self, x):
#         return self.block(x)


# class SimpleDetectionModel(nn.Module):
#     def __init__(
#         self,
#         nbr_classes: int,
#         kernel_size: int = 3,
#         model_name: str = "SimpleDetectionModel",
#     ):
#         super().__init__()

#         # encoder
#         self.blk1 = ConvBlock(
#             input_channels=1,
#             output_channels=64,
#             kernel_size=kernel_size,
#         )
#         self.blk1_pool = nn.MaxPool2d(2, 2)
#         # self.blk1_pool = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=2)
#         self.blk2 = ConvBlock(
#             input_channels=64,
#             output_channels=128,
#             kernel_size=kernel_size,
#         )
#         self.blk2_pool = nn.MaxPool2d(2, 2)
#         # self.blk2_pool = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=2)
#         self.blk3 = ConvBlock(
#             input_channels=128,
#             output_channels=256,
#             kernel_size=kernel_size,
#         )
#         self.blk3_pool = nn.MaxPool2d(2, 2)
#         # self.blk3_pool = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=2)
#         self.blk4 = ConvBlock(
#             input_channels=256,
#             output_channels=512,
#             kernel_size=kernel_size,
#         )
#         self.blk4_pool = nn.MaxPool2d(2, 2)
#         # self.blk4_pool = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=2)

#         # classifier
#         self.cl_dense = nn.Linear(in_features=512, out_features=nbr_classes)
#         self.cl_dropout = nn.Dropout(0.2)
#         self.cl_actv = nn.Softmax()

#         # ######### some metadata
#         self.model_name = model_name

#     def forward(self, x):
#         # through the encoder
#         x = self.blk1(x)
#         x = self.blk1_pool(x)

#         x = self.blk2(x)
#         x = self.blk2_pool(x)

#         x = self.blk3(x)
#         x = self.blk3_pool(x)

#         x = self.blk4(x)
#         x = self.blk4_pool(x)

#         # through the classifier (but first average pooling)
#         x = x.mean(dim=(-2, -1))
#         x = self.cl_dropout(x)
#         x = self.cl_dense(x)
#         # x = self.cl_actv(x)

#         return x


# class ResNet18DetectionModel(nn.Module):
#     def __init__(
#         self,
#         nbr_classes: int,
#         model_name: str = "ResNet18DetectionModel",
#         ImageNet_weight: bool = False,
#         freeze_weight: bool = False,
#     ):
#         super().__init__()

#         self.mdoel_name = model_name

#         # ResNet34 encoder
#         if ImageNet_weight:
#             self.model = torchvision.models.resnet18(weights=None)

#             if freeze_weight:
#                 for param in self.model.parameters():
#                     param.requires_grad = False
#         else:
#             self.model = self.model = torchvision.models.resnet18(weights=None)

#         # change number of output classes
#         num_ftrs = self.model.fc.in_features
#         # Here the size of each output sample is set to 2.
#         # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
#         self.model.fc = nn.Linear(num_ftrs, 128)
#         self.cl_dropout = nn.Dropout(0.2)
#         self.last_fc = nn.Linear(128, nbr_classes)

#     def forward(self, x):
#         x = self.model(x)
#         x = self.cl_dropout(x)


"""
Script that defines the detection models used by the run_model_training_routine.py in the context
of the qMRI project and tumor detection
"""

import numpy as np
from typing import Union
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    MaxPool2D,
    GlobalAveragePooling2D,
)

#         return self.last_fc(x)


def SimpleDetectionModel_TF(
    num_classes: int,
    input_shape: Union[list, tuple],
    class_weights: Union[list, tuple] = None,
    kernel_size: Union[list, tuple] = (3, 3),
    pool_size: Union[list, tuple] = (2, 2),
    model_name: str = "SimpleDetectionModel",
    debug: bool = False,
):

    if class_weights is None:
        class_weights = np.ones([1, num_classes])
    else:
        class_weights = class_weights

    # building  model
    input = Input(shape=input_shape)

    x = Conv2D(filters=64, kernel_size=kernel_size, activation="relu", padding="same")(
        input
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=kernel_size, activation="relu", padding="same")(
        x
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=pool_size)(x)

    x = Conv2D(filters=128, kernel_size=kernel_size, activation="relu", padding="same")(
        x
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=128, kernel_size=kernel_size, activation="relu", padding="same")(
        x
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=pool_size)(x)

    x = Conv2D(filters=256, kernel_size=kernel_size, activation="relu", padding="same")(
        x
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=256, kernel_size=kernel_size, activation="relu", padding="same")(
        x
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=pool_size)(x)

    x = Conv2D(filters=512, kernel_size=kernel_size, activation="relu", padding="same")(
        x
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=512, kernel_size=kernel_size, activation="relu", padding="same")(
        x
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=pool_size)(x)

    # x = encoder(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(
        units=128,
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    output = Dense(units=2, activation="softmax")(x)

    model = tf.keras.Model(inputs=input, outputs=output)

    # print model if needed
    if debug is True:
        print(model.summary())

    return model
