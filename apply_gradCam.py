# %%
"""
https://github.com/jacobgil/pytorch-grad-cam/
"""
import os
import pandas as pd
import random
import numpy as np
import PIL
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms

from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# local imports
import model_bucket_CBTN_v1

# %% GET PATH TO MODEL TO PLOT
FOLD_NAME = "fold_5"
PATH_TO_MODEL_CKTP = "/flush/iulta54/P5-PedMRI_CBTN_v1/trained_model_archive/TESTs_20231106/ResNet18_pretrained_True_frozen_True_0.4_LR_0.0001_BATCH_64_AUGMENTATION_True_OPTIM_sgd_SCHEDULER_exponential_MLPNODES_512_t0928/REPETITION_1/TB_fold_5/last.pt"
# MODEL_VERSION = os.path.basename(os.path.dirname(PATH_TO_MODEL_CKTP)).split("_")[0]
PATH_TO_SPLIT_CSV = "/flush/iulta54/P5-PedMRI_CBTN_v1/trained_model_archive/TESTs_20231106/ResNet18_pretrained_True_frozen_True_0.4_LR_0.0001_BATCH_64_AUGMENTATION_True_OPTIM_sgd_SCHEDULER_exponential_MLPNODES_512_t0928/REPETITION_1/data_split_information.csv"
CLASSES_TO_USE = ["ASTROCYTOMA", "EPENDYMOMA", "MEDULLOBLASTOMA"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% # ########################################### GET TRAINING, VALIDATION AND TEST FILES
print("Loading data...")
dataset_split = pd.read_csv(PATH_TO_SPLIT_CSV)
dataset_split = dataset_split.drop(columns=["level_0", "index"])

training_files = list(
    dataset_split.loc[
        (dataset_split[FOLD_NAME] == "training")
        & (dataset_split["target"].isin(CLASSES_TO_USE))
    ]["file_path"]
)
validation_files = list(
    dataset_split.loc[
        (dataset_split[FOLD_NAME] == "validation")
        & (dataset_split["target"].isin(CLASSES_TO_USE))
    ]["file_path"]
)
test_files = list(
    dataset_split.loc[
        (dataset_split[FOLD_NAME] == "test")
        & (dataset_split["target"].isin(CLASSES_TO_USE))
    ]["file_path"]
)

print(f"Nbr. training files: {len(training_files)}")
print(f"Nbr. validation file: {len(validation_files)}")
print(f"Nbr. test file: {len(test_files)}")


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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

train_transform = transforms.Compose(
    [
        # transforms.RandomResizedCrop(224, scale=(0.6, 1.5), ratio=(0.75, 1.33)),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        # transforms.RandomApply(
        #     [
        #         transforms.ColorJitter(
        #             brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
        #         )
        #     ],
        #     p=0.5,
        # ),
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([0.4451, 0.4262, 0.3959]), np.array([0.2411, 0.2403, 0.2466])
        ),
    ]
)

validation_transform = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([0.4451, 0.4262, 0.3959]), np.array([0.2411, 0.2403, 0.2466])
        ),
    ]
)

train_labels = list(
    dataset_split.loc[
        (dataset_split[FOLD_NAME] == "training")
        & (dataset_split["target"].isin(CLASSES_TO_USE))
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
    512,
    num_workers=15,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
)

# validation
validation_labels_numeric = [
    unique_labels.index(l)
    for l in list(
        dataset_split.loc[
            (dataset_split[FOLD_NAME] == "validation")
            & (dataset_split["target"].isin(CLASSES_TO_USE))
        ]["target"]
    )
]
validationdataset = PNGDatasetFromFolder(
    validation_files,
    transform=train_transform,
    labels=validation_labels_numeric,
)
validationloader = DataLoader(
    validationdataset,
    len(validation_files),
    num_workers=15,
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g,
)

# test
test_labels_numeric = [
    unique_labels.index(l)
    for l in list(
        dataset_split.loc[
            (dataset_split[FOLD_NAME] == "test")
            & (dataset_split["target"].isin(CLASSES_TO_USE))
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
    64,
    num_workers=15,
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g,
)

# %% LOAD MODEL
net = torch.load(PATH_TO_MODEL_CKTP)
net = net.model.to(device)
# %% APPLY gradCAM

target_layers = [net.model.layer4[-1].conv2]

dataiter = iter(validationloader)
images, labels = next(dataiter)

with torch.no_grad():
    pred = net(images.to(device))
    y_ = torch.argmax(pred, axis=1)

accuracy = np.sum(labels.numpy() == y_.to("cpu").numpy()) / labels.shape[0]
from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(labels.numpy(), y_.to("cpu").numpy())
print(f"{mcc}, {accuracy}")

# save confusion matrix
import torchmetrics
import seaborn as sns

confusion_matrix = torchmetrics.ConfusionMatrix(
    task="multiclass", num_classes=3, threshold=0.05
)
confusion_matrix(torch.tensor(labels.numpy()), torch.tensor(y_.to("cpu").numpy()))
confusion_matrix_computed = (
    confusion_matrix.compute().detach().cpu().numpy().astype(int)
)
df_cm = pd.DataFrame(confusion_matrix_computed)
fig = plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, annot=True, cmap="Blues", annot_kws={"size": 20})

cam = GradCAM(model=net, target_layers=target_layers, use_cuda=True)
targets = [
    ClassifierOutputTarget(0),
    ClassifierOutputTarget(1),
    ClassifierOutputTarget(2),
]

# compute GradCAM for all the images in the batch
grayscale_cam = cam(
    input_tensor=images, targets=None, aug_smooth=False, eigen_smooth=False
)
# %%
results = []
for i in range(images.shape[0]):
    rgb_img = np.transpose(images.numpy()[i], (1, 2, 0))
    # un-normalize
    rgb_img = rgb_img * np.array([0.2411, 0.2403, 0.2466]) + np.array(
        [0.4451, 0.4262, 0.3959]
    )
    # clip [0,1]
    rgb_img[rgb_img < 0] = 0
    rgb_img[rgb_img > 1] = 1
    visualization = show_cam_on_image(rgb_img, grayscale_cam[i, :], use_rgb=True)
    results.append(torch.tensor(visualization.transpose(2, 0, 1)))

fig = plt.figure(figsize=(30, 30))
plt.imshow(torchvision.utils.make_grid(results).numpy().transpose(1, 2, 0))
plt.colorbar()
# plt.imshow(PIL.Image.fromarray(np.hstack(results)))

# %%
