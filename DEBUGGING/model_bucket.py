import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


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


class Net_normalized(nn.Module):
    def __init__(self, nbr_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.norm1 = nn.InstanceNorm2d(num_features=6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.norm2 = nn.InstanceNorm2d(num_features=16)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.norm3 = nn.InstanceNorm1d(num_features=84)
        self.drop1 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(84, nbr_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.norm3(self.fc1(x)))
        x = self.drop1(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class Net(nn.Module):
    def __init__(self, nbr_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, nbr_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SDM4(torch.nn.Module):
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
            ), f"2D-SDM4 model build: the number of dropout rates and the number of encoder layers does not match. Given {len(conv_dropout_rate)} and {len(self.filters_in_encoder)}"
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
