# %%
"""
Script that goes and checke the common aspect that needs to be taken into consideration when performing image classification.
It will start with investigating the data: range values, normalization, distribution between training validation and test sets.
Then, it fill investigate the data generator itself: intensity range.
Following, a pytorch notebook for image classification is followed: fisrt on the given dataset and then on the custom dataset.
"""
# %% IMPORTS

import os
import glob
import pandas as pd
import pathlib
import numpy as np
import PIL
import seaborn as sns
import matplotlib.pyplot as plt
import random
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

from sklearn.utils import class_weight
from sklearn.metrics import matthews_corrcoef

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


import torchvision
import torchvision.transforms as transforms

import torchmetrics

import pytorch_lightning as pl

# %% CNN MODEL DEFINITIONS


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


# %% DEFINE CONFIGURATIONS
@hydra.main(version_base=None, config_path="conf", config_name="debug_conf_TCGA")
def main(cfg: DictConfig) -> None:
    set_up(cfg)
    run_training(cfg)


def set_up(cfg: DictConfig) -> None:
    # set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_nbr)

    # Seed everything
    pl.seed_everything(cfg.seed)
    torch.manual_seed(cfg.seed)
    # torch.use_deterministic_algorithms(True)

    # set os envoironment for deterministic
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # get number of workers
    cfg.nbr_workers = os.cpu_count()

    # Save starting time
    start_time_year_month_day = datetime.now().strftime("%y%m%d")
    start_time_hour_minutes = datetime.now().strftime("%H%M")
    # Set save path
    aus_name = f'{cfg.model_version}_opt_{cfg.optimizer}_sch_{cfg.scheduler}_lr_{cfg.base_lr}_fold_{cfg.fold_name}_expdec_{cfg.exponential_decay if cfg.scheduler == "EXPONENTIAL" else None}_{start_time_hour_minutes}'
    cfg.save_path = os.path.join(
        cfg.root_save_path,
        f"TESTs_TCGA_{start_time_year_month_day}",
        aus_name,
    )
    pathlib.Path(cfg.save_path).mkdir(parents=True, exist_ok=True)


def run_training(cfg: DictConfig) -> None:
    # ########################################### GET TRAINING, VALIDATION AND TEST FILES
    print("Loading data...")
    dataset_split = pd.read_csv(cfg.path_to_dataset_split)
    dataset_split = dataset_split.drop(columns=["level_0", "index"])

    training_files = list(
        dataset_split.loc[
            (dataset_split[cfg.fold_name] == "training")
            & (dataset_split["target"].isin(cfg.classes_to_use))
        ]["file_path"]
    )
    validation_files = list(
        dataset_split.loc[
            (dataset_split[cfg.fold_name] == "validation")
            & (dataset_split["target"].isin(cfg.classes_to_use))
        ]["file_path"]
    )
    test_files = list(
        dataset_split.loc[
            (dataset_split[cfg.fold_name] == "test")
            & (dataset_split["target"].isin(cfg.classes_to_use))
        ]["file_path"]
    )

    print(f"Nbr. training files: {len(training_files)}")
    print(f"Nbr. validation file: {len(validation_files)}")
    print(f"Nbr. test file: {len(test_files)}")

    # ########################################### DEFINE PRE-PROCESSING

    class PNGDatasetFromFolder(Dataset):
        def __init__(
            self,
            item_list,
            transform=None,
            labels=None,
        ):
            super().__init__()
            self.item_list = item_list
            self.nbr_total_imgs = len(self.item_list)
            self.transform = transform
            self.labels = labels

        def __len__(
            self,
        ):
            return self.nbr_total_imgs

        def __getitem__(self, item_index):
            item_path = self.item_list[item_index]
            image = PIL.Image.open(item_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = torchvision.transforms.functional.pil_to_tensor(image)

            # return label as well
            if self.labels:
                label = self.labels[item_index]
                return image, label
            else:
                return image, 0

    # ########################################### DEFINE DATALOADERS
    print("Creating data generators...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # reproducibility
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.6, 1.5), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                    )
                ],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_labels = list(
        dataset_split.loc[
            (dataset_split[cfg.fold_name] == "training")
            & (dataset_split["target"].isin(cfg.classes_to_use))
        ]["target"]
    )
    unique_labels = list(dict.fromkeys(train_labels))
    unique_labels.sort()

    # training
    train_labels_numeric = [unique_labels.index(l) for l in train_labels]
    trainset = PNGDatasetFromFolder(
        training_files,
        transform=train_transform,
        labels=train_labels_numeric,
    )
    trainloader = DataLoader(
        trainset,
        cfg.batch_size,
        num_workers=cfg.nbr_workers,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # validation
    validation_labels_numeric = [
        unique_labels.index(l)
        for l in list(
            dataset_split.loc[
                (dataset_split[cfg.fold_name] == "validation")
                & (dataset_split["target"].isin(cfg.classes_to_use))
            ]["target"]
        )
    ]
    validationdataset = PNGDatasetFromFolder(
        validation_files,
        transform=validation_transform,
        labels=validation_labels_numeric,
    )
    validationloader = DataLoader(
        validationdataset,
        cfg.batch_size,
        num_workers=cfg.nbr_workers,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # test
    test_labels_numeric = [
        unique_labels.index(l)
        for l in list(
            dataset_split.loc[
                (dataset_split[cfg.fold_name] == "test")
                & (dataset_split["target"].isin(cfg.classes_to_use))
            ]["target"]
        )
    ]

    testdataset = PNGDatasetFromFolder(
        test_files,
        transform=validation_transform,
        labels=test_labels_numeric,
    )
    testloader = DataLoader(
        testdataset,
        cfg.batch_size,
        num_workers=cfg.nbr_workers,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    classes = cfg.classes_to_use

    print("Computing class weights...")
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels_numeric),
        y=train_labels_numeric,
    )
    cfg.class_weights = class_weights.tolist()
    print(f"Class weights: {class_weights}")

    # save example training images
    print("Saving example training images...")

    def imshow(
        img,
        save_path: str = None,
        prefix: str = "Example training images",
        draw: bool = False,
    ):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        fig = plt.figure()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        if save_path:
            # save image
            fig.savefig(os.path.join(save_path, prefix))
            plt.close()
        else:
            if draw:
                plt.show()
            else:
                plt.close()

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images), save_path=cfg.save_path, draw=False)

    # ########################################### DEFINE CNN
    print(f"Building model ({cfg.model_version})...")

    if cfg.model_version.lower() == "resnet9":
        net = ResNet9(nbr_classes=len(cfg.classes_to_use))
    elif cfg.model_version.lower() == 'sdm4':
        net = SDM4(nbr_classes=len(cfg.classes_to_use))
    elif cfg.model_version.lower() == "bare_bone_cnn":
        net = Net(nbr_classes=len(cfg.classes_to_use))
    elif cfg.model_version.lower() == "bare_bone_cnn_normalized":
        net = Net_normalized(nbr_classes=len(cfg.classes_to_use))
    else:
        raise ValueError(
            f"The given model_version is not among the ones supported. Given {cfg.model_version}"
        )
    net.to(device)

    # ########################################### DEFINE CRITERION, OPTIMIZER AND LR SCHEDULER
    print("Defining criterion (weighted CCE)...")
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
    print(f"Defining optimizer ({cfg.optimizer})...")
    if cfg.optimizer.lower() == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=cfg.base_lr, momentum=0.9)
    elif cfg.optimizer.lower() == "adam":
        optimizer = optim.Adam(net.parameters(), lr=cfg.base_lr)
    else:
        raise ValueError(
            f"The given optimizer scheduler is not among the ones supported. Given {cfg.optimizer}"
        )

    print(f"Defining learning rate scheduler ({cfg.scheduler})...")
    if cfg.scheduler.lower() == "constant":
        scheduler = None
    elif cfg.scheduler.lower() == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer, start_factor=1, total_iters=cfg.max_epochs
        )
    elif cfg.scheduler.lower() == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.exponential_decay,
        )
    else:
        raise ValueError(
            f"The given learning rate scheduler is not among the ones supported. Given {cfg.scheduler}"
        )

    # ########################################### RUN TRAINING LOOP
    train_loss_history = []
    validation_loss_history = []

    print("Save configuration right before starting training...")
    OmegaConf.save(cfg, os.path.join(cfg.save_path, "hydra_config.yaml"))

    # validationloader = trainloader
    start_time = datetime.now()
    print(f"Starting training ({datetime.now().strftime('h%H-m%M-s%S')})")
    for epoch in range(cfg.max_epochs):  # loop over the dataset multiple times
        training_running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # update optimizer
            optimizer.step()

            # update LR scheduler
            if scheduler:
                scheduler.step()

            # update loss
            training_running_loss += loss.item()

        # save info
        train_loss_history.append(training_running_loss / len(trainloader))
        training_running_loss = 0.0

        with torch.no_grad():
            validation_running_loss = 0.0
            for i, data in enumerate(validationloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # update loss
                validation_running_loss += loss.item()

            # save stats
            validation_loss_history.append(
                validation_running_loss / len(validationloader)
            )
            validation_running_loss = 0.0

        # print stats
        print(
            f"Epoch [{epoch + 1:04d}\{cfg.max_epochs:0{len(str(cfg.max_epochs))}d}] (elapsed: {(datetime.now()-start_time)}) - Tr loss: {train_loss_history[-1]:.3f} - Val loss: {validation_loss_history[-1]:.3f} - LR: {scheduler.get_last_lr()[0] if scheduler else cfg.base_lr}"
        )

    print(f"Finished Training! (elapsed: {(datetime.now()-start_time)})")
    # % save model
    print("Saving model...")
    PATH = os.path.join(cfg.save_path, "last_model.pth")
    torch.save(net.state_dict(), PATH)

    ## plot training loss
    print("Saving training curves...")
    fig = plt.figure()
    plt.plot(train_loss_history, label="Training loss")
    plt.plot(validation_loss_history, label="Validation loss")
    plt.legend()
    fig.savefig(os.path.join(cfg.save_path, "training_curves.png"))
    plt.close(fig)

    print("Saving learning rate graph...")
    # # plot learning rate (need to reititialize scheduler)
    if scheduler:
        if cfg.optimizer.lower() == "sgd":
            optimizer = optim.SGD(net.parameters(), lr=cfg.base_lr, momentum=0.9)
        elif cfg.optimizer.lower() == "adam":
            optimizer = optim.Adam(net.parameters(), lr=cfg.base_lr)
        lrs = []
        steps = []
        for epoch in range(cfg.max_epochs):
            for batch in range(len(trainloader)):
                scheduler.step()
                lrs.append(scheduler.get_last_lr()[0])
                steps.append(epoch * len(trainloader) + batch)

        fig = plt.figure()
        plt.plot(steps, lrs, label="OneCycle")
        plt.legend()
        fig.savefig(os.path.join(cfg.save_path, "learning_rate.png"))
        plt.close(fig)

    # % ########################################### EVALUATE ON VALIDATION SET
    print("Running evaluation on validation set...")
    evaluation_logs = []

    # get predictions
    y = []
    pred = []

    correct = 0
    total = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in validationloader:
            images, labels = data[0].to(device), data[1].to(device)
            labels = labels.to("cpu")
            y.append(labels.detach().numpy())
            # calculate outputs by running images through the network
            outputs = net(images)
            outputs = outputs.to("cpu")
            pred.append(torch.argmax(outputs, 1).numpy())
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    log_str = str(
        f"Accuracy of the network on the validation images: {100 * correct // total} %"
    )
    print(log_str)
    evaluation_logs.append(log_str)
    # compute MCC
    mcc = matthews_corrcoef(np.hstack(y), np.hstack(pred))
    log_str = str(f"MCC of the network on the validation images: {mcc}")
    print(log_str)
    evaluation_logs.append(log_str)

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in validationloader:
            images, labels = data[0].to(device), data[1].to(device)
            labels = labels.to("cpu")
            outputs = net(images)
            outputs = outputs.to("cpu")
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[int(label)]] += 1
                total_pred[classes[int(label)]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        log_str = str(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")
        print(log_str)
        evaluation_logs.append(log_str)

    # save confusion matrix
    confusion_matrix = torchmetrics.ConfusionMatrix(
        task="multiclass", num_classes=len(cfg.classes_to_use), threshold=0.05
    )
    confusion_matrix(torch.tensor(np.hstack(y)), torch.tensor(np.hstack(pred)))
    confusion_matrix_computed = (
        confusion_matrix.compute().detach().cpu().numpy().astype(int)
    )
    df_cm = pd.DataFrame(confusion_matrix_computed)
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, cmap="Blues", annot_kws={"size": 20})
    fig.savefig(os.path.join(cfg.save_path, "validation_confusion_matrix.png"))
    plt.close(fig)

    # % ########################################### EVALUATE ON TEST SET
    print("Running evaluation on test set...")

    # get predictions
    y = []
    pred = []

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            labels = labels.to("cpu")
            y.append(labels.detach().numpy())
            # calculate outputs by running images through the network
            outputs = net(images)
            outputs = outputs.to("cpu")
            pred.append(torch.argmax(outputs, 1).numpy())
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    log_str = str(
        f"Accuracy of the network on the test images: {100 * correct // total} %"
    )
    print(log_str)
    evaluation_logs.append(log_str)
    # compute MCC
    mcc = matthews_corrcoef(np.hstack(y), np.hstack(pred))
    log_str = str(f"MCC of the network on the test images: {mcc}")
    print(log_str)
    evaluation_logs.append(log_str)

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            labels = labels.to("cpu")
            outputs = net(images)
            outputs = outputs.to("cpu")
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[int(label)]] += 1
                total_pred[classes[int(label)]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        log_str = str(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")
        print(log_str)
        evaluation_logs.append(log_str)

    # save confusion matrix
    confusion_matrix = torchmetrics.ConfusionMatrix(
        task="multiclass", num_classes=len(cfg.classes_to_use), threshold=0.05
    )
    confusion_matrix(torch.tensor(np.hstack(y)), torch.tensor(np.hstack(pred)))
    confusion_matrix_computed = (
        confusion_matrix.compute().detach().cpu().numpy().astype(int)
    )
    df_cm = pd.DataFrame(confusion_matrix_computed)
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, cmap="Blues", annot_kws={"size": 20})
    fig.savefig(os.path.join(cfg.save_path, "test_confusion_matrix.png"))
    plt.close(fig)

    # save test logs to file
    print("Saving test logs to file...")
    f = open(os.path.join(cfg.save_path, "test_logs.txt"), "w")
    for l in evaluation_logs:
        f.write(l + "\n")
    f.close()

    print("Script completed.")


if __name__ == "__main__":
    main()

# # %% 3##########################################################################


# # %% TRY ON TUMOR VS NON TUMOR CLASSIFICATION
# # reproducibility
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


# g = torch.Generator()
# g.manual_seed(0)

# train_transform = transforms.Compose(
#     [
#         transforms.RandomResizedCrop(224, scale=(0.6, 1.5), ratio=(0.75, 1.33)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(90),
#         transforms.RandomApply(
#             [
#                 transforms.ColorJitter(
#                     brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
#                 )
#             ],
#             p=0.5,
#         ),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ]
# )

# validation_transform = transforms.Compose(
#     [
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ]
# )

# cfg.batch_size = 64


# # training
# trainset = torchvision.datasets.ImageFolder(
#     root="/flush3/iulta54/Data/Dataset/not_shuffled/train",
#     transform=train_transform,
# )
# trainloader = DataLoader(
#     trainset,
#     cfg.batch_size,
#     num_workers=NUM_WORKERS,
#     shuffle=True,
#     worker_init_fn=seed_worker,
#     generator=g,
# )

# # validation
# validationdataset = torchvision.datasets.ImageFolder(
#     root="/flush3/iulta54/Data/Dataset/not_shuffled/val",
#     transform=validation_transform,
# )
# validationloader = DataLoader(
#     validationdataset,
#     cfg.batch_size,
#     num_workers=NUM_WORKERS,
#     shuffle=False,
#     worker_init_fn=seed_worker,
#     generator=g,
# )

# # test
# testdataset = torchvision.datasets.ImageFolder(
#     root="/flush3/iulta54/Data/Dataset/not_shuffled/test",
#     transform=validation_transform,
# )

# testloader = DataLoader(
#     testdataset,
#     cfg.batch_size,
#     num_workers=NUM_WORKERS,
#     shuffle=False,
#     worker_init_fn=seed_worker,
#     generator=g,
# )

# classes = []

# # ## show some figures
# import matplotlib.pyplot as plt
# import numpy as np


# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


# classes = ["without_tumor", "with_tumor"]

# # get some random training images
# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# # show images
# imshow(torchvision.utils.make_grid(images))

# # %% ## define CNN
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def conv_block(in_channels, out_channels, pool=False):
#     layers = [
#         nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True),
#     ]
#     if pool:
#         layers.append(nn.MaxPool2d(2))
#     return nn.Sequential(*layers)


# class ResNet9(nn.Module):
#     def __init__(self, nbr_classes: int = 10, in_channels: int = 3):
#         super().__init__()

#         self.conv1 = conv_block(in_channels, 64)
#         self.conv2 = conv_block(64, 128, pool=True)
#         self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

#         self.conv3 = conv_block(128, 256, pool=True)
#         self.conv4 = conv_block(256, 512, pool=True)
#         self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

#         self.classifier = nn.Sequential(
#             nn.AdaptiveMaxPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Dropout(0.2),
#             nn.Linear(512, nbr_classes),
#         )

#     def forward(self, xb):
#         out = self.conv1(xb)
#         out = self.conv2(out)
#         out = self.res1(out) + out
#         out = self.conv3(out)
#         out = self.conv4(out)
#         out = self.res2(out) + out
#         out = self.classifier(out)
#         return out


# # class Net(nn.Module):
# #     def __init__(self, nbr_classes: int = 3):
# #         super().__init__()
# #         self.conv1 = nn.Conv2d(3, 6, 5)
# #         self.norm1 = nn.InstanceNorm2d(num_features=6)
# #         self.pool = nn.MaxPool2d(2, 2)
# #         self.conv2 = nn.Conv2d(6, 16, 5)
# #         self.norm2 = nn.InstanceNorm2d(num_features=16)
# #         self.fc1 = nn.Linear(16 * 53 * 53, 120)
# #         self.fc2 = nn.Linear(120, 84)
# #         self.norm3 = nn.InstanceNorm1d(num_features=84)
# #         self.drop1 = nn.Dropout(0.2)
# #         self.fc3 = nn.Linear(84, nbr_classes)

# #     def forward(self, x):
# #         x = self.pool(F.relu(self.norm1(self.conv1(x))))
# #         x = self.pool(F.relu(self.norm2(self.conv2(x))))
# #         x = torch.flatten(x, 1)  # flatten all dimensions except batch
# #         x = F.relu(self.norm3(self.fc1(x)))
# #         x = self.drop1(F.relu(self.fc2(x)))
# #         x = self.fc3(x)
# #         return x


# class Net(nn.Module):
#     def __init__(self, nbr_classes: int = 3):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 53 * 53, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, nbr_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)  # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# # net = ResNet9(nbr_classes=len(cfg.classes_to_use))
# net = Net(nbr_classes=len(classes))
# net.to(device)

# # ## define loss function and optimizer
# class_weights = class_weight.compute_class_weight(
#     class_weight="balanced",
#     classes=np.unique(trainset.targets),
#     y=trainset.targets,
# )

# max_epochs = 10
# base_LR = 0.00001
# max_LR = 0.01
# criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
# # optimizer = optim.SGD(net.parameters(), lr=base_LR, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=base_LR)
# scheduler = None
# scheduler = torch.optim.lr_scheduler.ExponentialLR(
#     optimizer,
#     gamma=0.998,
# )
# # scheduler = torch.optim.lr_scheduler.OneCycleLR(
# #     optimizer, max_lr=max_LR, steps_per_epoch=len(trainloader), epochs=max_epochs
# # )

# # %% train network
# train_loss_history = []
# validation_loss_history = []
# training_mcc_history = []
# validation_mcc_history = []

# # validationloader = trainloader
# print("Starting training")
# for epoch in range(max_epochs):  # loop over the dataset multiple times
#     training_running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data[0].to(device), data[1].to(device)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()

#         # update optimizer
#         optimizer.step()

#         # update LR scheduler
#         if scheduler:
#             scheduler.step()

#         # update loss
#         training_running_loss += loss.item()

#     # save info
#     train_loss_history.append(training_running_loss / len(trainloader))
#     training_running_loss = 0.0

#     with torch.no_grad():
#         validation_running_loss = 0.0
#         for i, data in enumerate(validationloader, 0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data[0].to(device), data[1].to(device)

#             # forward + backward + optimize
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)

#             # update loss
#             validation_running_loss += loss.item()

#         # save stats
#         validation_loss_history.append(validation_running_loss / len(validationloader))
#         validation_running_loss = 0.0

#     # print stats
#     print(
#         f"Epoch [{epoch + 1:04d}] - Tr loss: {train_loss_history[-1]:.3f} - Val loss: {validation_loss_history[-1]:.3f} - LR: {scheduler.get_last_lr()[0]}"
#     )

# print("Finished Training")

# # %% save model

# PATH = os.path.join(SAVE_PATH, "NET_TUmor_vs_NON_TUMOR_model.pth")
# torch.save(net.state_dict(), PATH)

# ## plot training loss
# fig = plt.figure()
# plt.plot(train_loss_history, label="Training loss")
# plt.plot(validation_loss_history, label="Validation loss")
# plt.legend()
# fig.savefig(os.path.join(SAVE_PATH, "NET_TUmor_vs_NON_TUMOR_model.png"))

# # %% ## run test
# dataiter = iter(testloader)
# images, labels = next(dataiter)

# # # print images
# # imshow(torchvision.utils.make_grid(images))
# # print("GroundTruth: ", " ".join(f"{classes[labels[j]]:5s}" for j in range(4)))

# # net = ResNet9(nbr_classes=len(cfg.classes_to_use))
# net = Net(nbr_classes=len(classes))
# net.load_state_dict(torch.load(PATH))

# # get predictions
# y = []
# pred = []

# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         y.append(labels.detach().numpy())
#         # calculate outputs by running images through the network
#         outputs = net(images)
#         pred.append(torch.argmax(outputs, 1).numpy())
#         # the class with the highest energy is what we choose as prediction
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Accuracy of the network on the test images: {100 * correct // total} %")
# mcc = matthews_corrcoef(np.hstack(y), np.hstack(pred))
# print(f"MCC of the network on the test images: {mcc}")

# # prepare to count predictions for each class
# correct_pred = {classname: 0 for classname in classes}
# total_pred = {classname: 0 for classname in classes}

# # again no gradients needed
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predictions = torch.max(outputs, 1)
#         # collect the correct predictions for each class
#         for label, prediction in zip(labels, predictions):
#             if label == prediction:
#                 correct_pred[classes[label]] += 1
#             total_pred[classes[label]] += 1


# # print accuracy for each class
# for classname, correct_count in correct_pred.items():
#     accuracy = 100 * float(correct_count) / total_pred[classname]
#     print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")


# # %% FOR EACH OF THE DATASETS, PLOT THE HISTOGRAM DISTRIBUTION
# # for ds_files, ds_name in zip(
# #     [training_files, validation_files, test_files], ["training", "validation", "test"]
# # ):
# for ds_files, ds_name in zip([validation_files, test_files], ["validation", "test"]):
#     ds = PNGDatasetFromFolder(
#         ds_files,
#         transform=None,
#         labels=None,
#     )

#     ds_datasloader = DataLoader(
#         ds,
#         1,
#         num_workers=NUM_WORKERS,
#         shuffle=False,
#     )

#     # get all the elements in the dataloader
#     values = []
#     set = iter(ds_datasloader)
#     for idx, (x, y) in enumerate(set):
#         x = x.numpy()[:, 0, :, :]
#         values.append(x.reshape(x.shape[0], -1))
#     # stach all elements
#     values = np.vstack(values)

#     # start plotting
#     print(f"Plotting histogram for {ds_name} set.")
#     fig, axis = plt.subplots(nrows=1, ncols=1)
#     for i in range(values.shape[0]):
#         print(f"Worning on {i+1}\{values.shape[0]} line. \r", end="")
#         # plot a line histogram for each element
#         value = values[i, :]
#         sns.kdeplot(value, alpha=0.5)
#         # density = stats.gaussian_kde(value)
#         # num_bins = 50
#         # n, x, _ = plt.hist(value, num_bins, density=True, histtype="step")
#         # axis.plot(x, density(x), alpha=0.5)
#         axis.set_ylim([1e-5, 1e1])
#         axis.set_yscale("log")
#         # if i == 9:
#         #     break

#     # save figure
#     print("\nSaving figure...")
#     fig.savefig(os.path.join(SAVE_PATH, f"Histogram_{ds_name}_set.png"))
#     print("Done!")
# # %% PLOT THE SAME WITH THE USUAL PREPROCESSING

# img_mean = np.array([0.5, 0.5, 0.5])
# img_std = np.array([0.5, 0.5, 0.5])

# preprocess = transforms.Compose(
#     [
#         transforms.Resize(size=[224, 224], antialias=True),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=img_mean,
#             std=img_std,
#         ),
#     ],
# )

# for ds_files, ds_name in zip(
#     [training_files, validation_files, test_files], ["training", "validation", "test"]
# ):
#     ds = PNGDatasetFromFolder(
#         ds_files,
#         transform=preprocess,
#         labels=None,
#     )

#     ds_datasloader = DataLoader(
#         ds,
#         1,
#         num_workers=NUM_WORKERS,
#         shuffle=False,
#     )

#     # get all the elements in the dataloader
#     values = []
#     set = iter(ds_datasloader)
#     for idx, (x, y) in enumerate(set):
#         x = x.numpy()[:, 0, :, :]
#         values.append(x.reshape(x.shape[0], -1))
#     # stach all elements
#     values = np.vstack(values)

#     # start plotting
#     print(f"Plotting histogram for {ds_name} set.")
#     fig, axis = plt.subplots(nrows=1, ncols=1)
#     for i in range(values.shape[0]):
#         print(f"Worning on {i+1}\{values.shape[0]} line. \r", end="")
#         # plot a line histogram for each element
#         value = values[i, :]
#         sns.kdeplot(value, alpha=0.5)
#         # density = stats.gaussian_kde(value)
#         # num_bins = 50
#         # n, x, _ = plt.hist(value, num_bins, density=True, histtype="step")
#         # axis.plot(x, density(x), alpha=0.5)
#         axis.set_ylim([1e-5, 1e1])
#         axis.set_yscale("log")
#         if i == 100:
#             break

#     # save figure
#     print("\nSaving figure...")
#     fig.savefig(
#         os.path.join(SAVE_PATH, f"Histogram_{ds_name}_set_With_preprocessing.png")
#     )
#     print("Done!")

# # %% IMPLEMENT CLASSIFIER USING PYTORCH TUTORIAL - USE GIVEN DATASET
# # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
# )

# cfg.batch_size = 4

# trainset = torchvision.datasets.CIFAR10(
#     root="./data", train=True, download=True, transform=transform
# )
# trainloader = torch.utils.data.DataLoader(
#     trainset, cfg.batch_size=cfg.batch_size, shuffle=True, num_workers=2
# )

# testset = torchvision.datasets.CIFAR10(
#     root="./data", train=False, download=True, transform=transform
# )
# testloader = torch.utils.data.DataLoader(
#     testset, cfg.batch_size=cfg.batch_size, shuffle=False, num_workers=2
# )

# classes = (
#     "plane",
#     "car",
#     "bird",
#     "cat",
#     "deer",
#     "dog",
#     "frog",
#     "horse",
#     "ship",
#     "truck",
# )

# # ## show some figures
# import matplotlib.pyplot as plt
# import numpy as np


# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


# # get some random training images
# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(" ".join(f"{classes[labels[j]]:5s}" for j in range(cfg.batch_size)))


# # ## define CNN
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)  # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# net = Net()
# net.to(device)

# # ## define loss function and optimizer
# import torch.optim as optim

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# # ## train network
# loss_history = []
# max_epochs = 10
# for epoch in range(10):  # loop over the dataset multiple times
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data[0].to(device), data[1].to(device)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:  # print every 2000 mini-batches
#             print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
#             loss_history.append(running_loss)
#             running_loss = 0.0

# print("Finished Training")

# # ## save model
# PATH = os.path.join(SAVE_PATH, "ImageNet_trained_model.pth")
# torch.save(net.state_dict(), PATH)

# # ## plot training loss
# plt.plot(loss_history)

# # %% ## run test
# dataiter = iter(testloader)
# images, labels = next(dataiter)

# # print images
# imshow(torchvision.utils.make_grid(images))
# print("GroundTruth: ", " ".join(f"{classes[labels[j]]:5s}" for j in range(4)))

# net = Net()
# net.load_state_dict(torch.load(PATH))

# outputs = net(images)
# _, predicted = torch.max(outputs, 1)

# print("Predicted: ", " ".join(f"{classes[predicted[j]]:5s}" for j in range(4)))

# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         # calculate outputs by running images through the network
#         outputs = net(images)
#         # the class with the highest energy is what we choose as prediction
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")

# # prepare to count predictions for each class
# correct_pred = {classname: 0 for classname in classes}
# total_pred = {classname: 0 for classname in classes}

# # again no gradients needed
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predictions = torch.max(outputs, 1)
#         # collect the correct predictions for each class
#         for label, prediction in zip(labels, predictions):
#             if label == prediction:
#                 correct_pred[classes[label]] += 1
#             total_pred[classes[label]] += 1


# # print accuracy for each class
# for classname, correct_count in correct_pred.items():
#     accuracy = 100 * float(correct_count) / total_pred[classname]
#     print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")
