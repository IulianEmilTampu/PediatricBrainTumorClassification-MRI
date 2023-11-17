# %% SOURCE
"""
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
and
https://github.com/paulhager/MMCL-Tabular-Imaging/tree/main
"""
# %% IMPORTS

## Standard libraries
import os
from copy import deepcopy

## Imports for plotting
import matplotlib.pyplot as plt
import PIL
import pandas as pd
import numpy as np
from datetime import datetime

plt.set_cmap("cividis")
import matplotlib

matplotlib.rcParams["lines.linewidth"] = 2.0
import seaborn as sns

sns.set()

## tqdm for loading bars
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

## Trochmetrics
import torchmetrics

## Torchvision
import torchvision
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "./data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = f"./saved_models/CBTN/IET_test_{datetime.now().strftime('%Y%m%d')}_{datetime.now().strftime('t%H%M')}"
# In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
# workers as possible in a data loader, which corresponds to the number of CPU cores
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
NUM_WORKERS = os.cpu_count()

# Setting the seed
pl.seed_everything(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)


# %% DEFINE CBTN-like contrastive dataset


class ContrastiveTransformations(object):
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


class PNGDatasetFromFolder(Dataset):
    def __init__(self, item_list, transform, labels=None, return_file_path=False):
        super().__init__()
        self.item_list = item_list
        self.nbr_total_imgs = len(self.item_list)
        self.transform = transform
        self.labels = labels
        self.return_file_path = return_file_path

    def __len__(
        self,
    ):
        return self.nbr_total_imgs

    def __getitem__(self, item_index):
        item_path = self.item_list[item_index]
        image = PIL.Image.open(item_path).convert("RGB")
        tensor_image = self.transform(image)
        # return label as well
        if self.labels:
            label = self.labels[item_index]
            if self.return_file_path:
                return tensor_image, label, item_path
            else:
                return tensor_image, label
        else:
            if self.return_file_path:
                return tensor_image, 0, item_path
            else:
                return tensor_image, 0


class SimCLRCustomDataset(pl.LightningDataModule):
    def __init__(
        self,
        train_sample_paths,
        batch_size,
        num_workers,
        validation_sample_paths=None,
        train_val_ratio=0.8,
        test_sample_paths=None,
        preprocess=None,
        transforms=None,
        heuristic_for_class_extraction=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_sample_paths = train_sample_paths
        self.validation_sample_paths = validation_sample_paths
        self.test_sample_paths = test_sample_paths
        if isinstance(self.test_sample_paths, (list, tuple)):
            self.return_test_dataloader = True
            print("Returning testing dataloader")
        else:
            self.return_test_dataloader = False
        self.preprocess = preprocess
        self.transform = transforms
        self.num_workers = num_workers
        self.train_val_ratio = train_val_ratio
        self.return_classes = True if heuristic_for_class_extraction else False
        # TODO make it work even when only train_sample_paths is given

        # get classes
        if self.return_classes:
            self.train_per_sample_target_class = [
                heuristic_for_class_extraction(str(f)) for f in self.train_sample_paths
            ]
            self.validation_per_sample_target_class = [
                heuristic_for_class_extraction(str(f))
                for f in self.validation_sample_paths
            ]
            if self.return_test_dataloader:
                self.test_per_sample_target_class = [
                    heuristic_for_class_extraction(str(f))
                    for f in self.test_sample_paths
                ]

            # get unique target classes
            unique_targe_classes = dict.fromkeys(self.train_per_sample_target_class)
            one_hot_encodings = torch.nn.functional.one_hot(
                torch.tensor(list(range(len(unique_targe_classes))))
            )
            one_hot_encodings = [i.type(torch.float32) for i in one_hot_encodings]
            # build mapping between class and one hot encoding
            self.target_class_to_one_hot_mapping = dict(
                zip(unique_targe_classes, one_hot_encodings)
            )

            # make one hot encodings
            self.train_per_sample_target_class = [
                self.target_class_to_one_hot_mapping[c]
                for c in self.train_per_sample_target_class
            ]
            self.validation_per_sample_target_class = [
                self.target_class_to_one_hot_mapping[c]
                for c in self.validation_per_sample_target_class
            ]
            if self.return_test_dataloader:
                self.test_per_sample_target_class = [
                    self.target_class_to_one_hot_mapping[c]
                    for c in self.test_per_sample_target_class
                ]
        else:
            self.train_per_sample_target_class = None
            self.validation_per_sample_target_class = None
            self.test_per_sample_target_class = None

    def prepare_data(self):
        print("Working on preparing the data")
        return

    def setup(self, stage=None):
        self.training_set = PNGDatasetFromFolder(
            self.train_sample_paths,
            transform=self.transform,
            labels=self.train_per_sample_target_class,
        )
        self.validation_set = PNGDatasetFromFolder(
            self.validation_sample_paths,
            transform=self.preprocess,
            labels=self.validation_per_sample_target_class,
        )

        if self.return_test_dataloader:
            self.test_set = PNGDatasetFromFolder(
                self.test_sample_paths,
                transform=self.preprocess,
                labels=self.test_per_sample_target_class,
            )

    def train_dataloader(self):
        return DataLoader(
            self.training_set,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_set,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        if self.return_test_dataloader:
            return DataLoader(
                self.test_set,
                self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )

    def get_class_weights(self):
        if self.return_classes:
            # compute class weights based on the training set
            training_target_classes = [
                v.numpy().argmax() for v in self.train_per_sample_target_class
            ]
            class_ratios = class_weight.compute_class_weight(
                class_weight="balanced",
                classes=np.unique(training_target_classes),
                y=training_target_classes,
            )
            return class_ratios
        else:
            return None


contrast_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(size=224),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
                )
            ],
            p=0.8,
        ),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# %% GET DATASET SPLIT
import dataset_utilities
import importlib

importlib.reload(dataset_utilities)

config = {
    "dataloader_settings": {
        "dataset_path": "/flush/iulta54/Data/CBTN_v1/T2_FILTERED_SLICES_RLP_20_80",
        "classes_of_interest": ["ASTROCYTOMA", "EPENDYMOMA", "MEDULLOBLASTOMA"],
        "test_size": 0.2,
        "class_stratification": True,
        "input_size": [224, 224],
    },
    "training_settings": {
        "random_state": 42,
        "nbr_inner_cv_folds": 1,
    },
    "debugging_settings": {"dataset_fraction": 1.0},
    "logging_settings": {
        "checkpoint_path": CHECKPOINT_PATH,
        "save_training_validation_test_hist": CHECKPOINT_PATH,
    },
}

# ## get splits based on the configuration
heuristic_for_file_discovery = "*.png"
heuristic_for_subject_ID_extraction = lambda x: os.path.basename(x).split("___")[2]
heuristic_for_class_extraction = lambda x: os.path.basename(x).split("___")[0]

dataset_split_df = dataset_utilities._get_split(
    config,
    heuristic_for_file_discovery=heuristic_for_file_discovery,
    heuristic_for_subject_ID_extraction=heuristic_for_subject_ID_extraction,
    heuristic_for_class_extraction=heuristic_for_class_extraction,
    repetition_number=1,
)

# %% BUILD DATA GENERATORS
print(" # Building data generators for SimCLR training...")
files_for_training = list(
    dataset_split_df.loc[dataset_split_df[f"fold_1"] == "training"]["file_path"]
)
files_for_validation = list(
    dataset_split_df.loc[dataset_split_df[f"fold_1"] == "validation"]["file_path"]
)
files_for_testing = list(
    dataset_split_df.loc[dataset_split_df[f"fold_1"] == "test"]["file_path"]
)
train_val_dataloader = SimCLRCustomDataset(
    train_sample_paths=files_for_training,
    validation_sample_paths=files_for_validation,
    test_sample_paths=files_for_testing,
    batch_size=128,
    num_workers=NUM_WORKERS,
    preprocess=ContrastiveTransformations(contrast_transforms, n_views=2),
    transforms=ContrastiveTransformations(contrast_transforms, n_views=2),
    heuristic_for_class_extraction=heuristic_for_class_extraction,
)
print(" # Done!")

# %% PLOT EXAMPLE IMAGES
print(" # Saving some example images...")
pl.seed_everything(42)
NUM_IMAGES = 6
train_val_dataloader.setup()
train_ds = train_val_dataloader.train_dataloader()
dataset = iter(train_ds)

# take out some images
xi, xj = [], []
for x, y in dataset:
    xi.extend([x[0][i, :, :, :] for i in range(x[0].shape[0])])
    xj.extend([x[1][i, :, :, :] for i in range(x[1].shape[0])])

    if len(xi) >= NUM_IMAGES:
        break

imgs = xi[0:NUM_IMAGES]
imgs.extend(xj[0:NUM_IMAGES])
# imgs = torch.stack((torch.stack(xi, dim=0), torch.stack(xj, dim=0)), dim=0)
img_grid = torchvision.utils.make_grid(
    imgs, nrow=NUM_IMAGES, normalize=True, pad_value=0.9
)
img_grid = img_grid.permute(1, 2, 0)

fig = plt.figure(figsize=(10, 5))
plt.title("Augmented image examples of the CBTN dataset")
plt.imshow(img_grid)
plt.axis("off")
# plt.show()
fig.savefig(os.path.join(CHECKPOINT_PATH, "Example_SimCLR_images.png"))
plt.close()

print(
    f' # Done! (saved at {os.path.join(CHECKPOINT_PATH, "Example_SimCLR_images.png")})'
)


# %% DEFINE MODEL
class SimCLR(pl.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert (
            self.hparams.temperature > 0.0
        ), "The temperature must be a positive float!"
        # Base model f(.)
        # self.convnet = torchvision.models.resnet18(
        #     num_classes=4 * hidden_dim
        # )  # Output of last linear layer
        self.convnet = torchvision.models.resnet50(
            num_classes=4 * hidden_dim
        )  # Output of last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
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
        self.log(mode + "_loss", nll)
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
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")


# %% DEFINE TRAINING TRAINER
def train_simclr(
    train_val_dataloader, max_epochs=10, use_pretrained=False, ckpt_path=None, **kwargs
):
    trainer = pl.Trainer(
        default_root_dir=os.path.join(
            CHECKPOINT_PATH, f"SimCLR_{datetime.now().strftime('t%H%M')}"
        ),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_top5"),
            LearningRateMonitor("epoch"),
        ],
        log_every_n_steps=1,
    )
    trainer.logger._default_hp_metric = (
        None  # Optional logging argument that we don't need
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    if use_pretrained:
        if os.path.isfile(ckpt_path):
            print(f"Found pretrained model at {ckpt_path}, loading...")
            model = SimCLR.load_from_checkpoint(
                ckpt_path
            )  # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(42)  # To be reproducable
        model = SimCLR(max_epochs=max_epochs, **kwargs)
        trainer.fit(model, train_val_dataloader)
        model = SimCLR.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )  # Load best checkpoint after training
        # save model
        # torch.save(
        #     model,
        #     os.path.join(trainer.checkpoint_callback.best_model_path, "best_model.pt"),
        # )

    return model


print(" # Start training Encoder using SimCLR framework...")
simclr_model = train_simclr(
    train_val_dataloader,
    use_pretrained=False,
    ckpt_path="/flush/iulta54/P5-PedMRI_CBTN_v1/saved_models/CBTN/IET_test_20231106/SimCLR_t1707/lightning_logs/version_0/checkpoints/epoch=440-step=15876.ckpt",
    hidden_dim=128,
    max_epochs=1000,
    lr=5e-4,
    temperature=0.07,
    weight_decay=1e-4,
)
print(" # Done!")

# %% PREPARE DATA FOR CLASSIFICATION
print(" # Preparing data for training the logistic regression model...")
# get mapping of the classes to one hot encoding
unique_targe_classes = dict.fromkeys(pd.unique(dataset_split_df["target"]))
one_hot_encodings = torch.nn.functional.one_hot(
    torch.tensor(list(range(len(unique_targe_classes))))
)
one_hot_encodings = [i.type(torch.float32) for i in one_hot_encodings]
# build mapping between class and one hot encoding
target_class_to_one_hot_mapping = dict(zip(unique_targe_classes, one_hot_encodings))

# train on the validation and test on the test - get labels to one hot encoding
training_samples = dataset_split_df.loc[dataset_split_df[f"fold_1"] == "training"]
validation_samples = dataset_split_df.loc[dataset_split_df[f"fold_1"] == "validation"]

train_labels = [
    target_class_to_one_hot_mapping[c] for c in list(training_samples["target"])
]
val_labels = [
    target_class_to_one_hot_mapping[c] for c in list(validation_samples["target"])
]

# define augmentations (different for training and validation)
train_img_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=240),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
                )
            ],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

val_img_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# build generators
batch_size = 64
train_img_data = DataLoader(
    dataset_utilities.PNGDatasetFromFolder(
        list(training_samples["file_path"]),
        transform=val_img_transforms,
        labels=train_labels,
        return_file_path=False,
    ),
    batch_size=batch_size,
    num_workers=NUM_WORKERS,
    shuffle=True,
    drop_last=False,
)

val_img_data = DataLoader(
    dataset_utilities.PNGDatasetFromFolder(
        list(validation_samples["file_path"]),
        transform=val_img_transforms,
        labels=val_labels,
        return_file_path=False,
    ),
    batch_size=batch_size,
    num_workers=NUM_WORKERS,
    shuffle=False,
    drop_last=False,
)

print(
    f"Number of training samples: {len(train_img_data.dataset)} (NOTE: more samples can be generated using augmentation)"
)
print("Number of test samples:", len(val_img_data.dataset))

# %% ENCODE ALL THE SAMPLES USIGN THE PRETRAINED MDOEL
print(" # Encoding images to features using the trained model...")

# this allows to get more 'data' through augmentation in the training set
# nbr_training_images_to_generate = len(list(training_samples["file_path"]))
nbr_training_images_to_generate = None


@torch.no_grad()
def prepare_data_features(model, data_loader, nbr_images_to_generate: int = None):
    # Prepare model
    network = deepcopy(model.convnet)
    network.fc = nn.Identity()  # Removing projection head g(.)
    network.eval()
    network.to(device)

    # # Encode as many images as requested
    feats, labels = [], []
    nbr_images_to_generate = (
        nbr_images_to_generate if nbr_images_to_generate else len(data_loader.dataset)
    )
    counter = 0
    while counter < nbr_images_to_generate:
        for idx, (batch_imgs, batch_labels) in enumerate(data_loader):
            print(f"Processing {counter+1}\{nbr_images_to_generate}\r", end="")
            batch_imgs = batch_imgs.to(device)
            batch_feats = network(batch_imgs)
            feats.append(batch_feats.detach().cpu())
            labels.append(batch_labels)

            # update counter
            counter += batch_labels.shape[0]
    print("\n")
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    # # Sort images by labels
    # labels, idxs = labels.sort()
    # feats = feats[idxs]

    return data.TensorDataset(feats, labels)


train_feats_simclr = prepare_data_features(
    simclr_model, train_img_data, nbr_training_images_to_generate
)
test_feats_simclr = prepare_data_features(simclr_model, val_img_data)
print(" # Done!")


# %% UTILITY TO REDUCE THE DATASET SIZE
def get_smaller_dataset(original_dataset, num_imgs):
    return data.TensorDataset(
        *[tensor[:num_imgs] for tensor in original_dataset.tensors]
    )


# %% DEFINE LOGISTIC REGRESSION MODEL - CLASSIFIER OVER THE EXTRACTED FEATURES
class LogisticRegression(pl.LightningModule):
    def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        # Mapping from representation h to classes
        self.model = nn.Linear(feature_dim, num_classes)

        # what to log during training, validation and testing
        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(
                    task="multiclass",
                    num_classes=num_classes,
                ),
                torchmetrics.MatthewsCorrCoef(
                    task="multiclass",
                    num_classes=num_classes,
                ),
            ]
        )
        self.train_metrics = metrics.clone(prefix="ptl/train_")
        self.valid_metrics = metrics.clone(prefix="ptl/val_")
        self.test_metrics = metrics.clone(prefix="ptl/test_")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # optimizer = torch.optim.SGD(
        #     self.model.parameters(),
        #     lr=self.hparams.lr,
        #     weight_decay=self.hparams.weight_decay,
        #     momentum=0.9,
        # )

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.99,
        )
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(
        #     optimizer,
        #     milestones=[
        #         int(self.hparams.max_epochs * 0.6),
        #         int(self.hparams.max_epochs * 0.8),
        #     ],
        #     gamma=0.1,
        # )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        feats, labels = batch
        preds = self.model(feats)
        loss = F.cross_entropy(preds, labels.argmax(dim=-1))

        # lof the other metrics
        if mode == "train":
            self.train_metrics.update(
                torch.argmax(preds, dim=-1), torch.argmax(labels, dim=-1)
            )
        elif mode == "val":
            self.valid_metrics.update(
                torch.argmax(preds, dim=-1), torch.argmax(labels, dim=-1)
            )
        elif mode == "test":
            self.test_metrics.update(
                torch.argmax(preds, dim=-1), torch.argmax(labels, dim=-1)
            )
        # acc = (preds.argmax(dim=-1) == labels.argmax(dim=-1)).float().mean()

        self.log("ptl/" + mode + "_loss", loss)
        # self.log(mode + "_acc", acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

    def on_train_epoch_end(
        self,
    ):
        # save metrics
        output = self.train_metrics.compute()
        self.log_dict(output)
        # reset
        self.train_metrics.reset()

    def on_validation_epoch_end(
        self,
    ):
        # save metrics
        output = self.valid_metrics.compute()
        self.log_dict(output)
        # reset
        self.valid_metrics.reset()

    def on_test_epoch_end(
        self,
    ):
        # save metrics
        output = self.test_metrics.compute()
        self.log_dict(output)
        # reset
        self.test_metrics.reset()


# %% DEFINE TRAINING FUNCTION
def train_logreg(
    train_feats_data,
    test_feats_data,
    model_suffix,
    max_epochs=100,
    batch_size=64,
    use_pretrained=False,
    **kwargs,
):
    trainer = pl.Trainer(
        default_root_dir=os.path.join(
            CHECKPOINT_PATH, f"LogisticRegression_{datetime.now().strftime('t%H%M')}"
        ),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="ptl/val_loss"),
            LearningRateMonitor("epoch"),
        ],
        enable_progress_bar=False,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
    )
    trainer.logger._default_hp_metric = None

    # Data loaders
    train_loader = data.DataLoader(
        train_feats_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=15,
    )
    test_loader = data.DataLoader(
        test_feats_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=15,
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    if use_pretrained:
        pretrained_filename = os.path.join(
            CHECKPOINT_PATH, f"LogisticRegression_{model_suffix}.ckpt"
        )
        if os.path.isfile(pretrained_filename):
            print(f"Found pretrained model at {pretrained_filename}, loading...")
            model = LogisticRegression.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = LogisticRegression(**kwargs)
        trainer.fit(model, train_loader, test_loader)
        model = LogisticRegression.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )
        # save model
        # torch.save(
        #     model,
        #     os.path.join(trainer.checkpoint_callback.best_model_path, "best_model.pt"),
        # )

    # Test best model on train and validation set
    train_result = trainer.test(model, train_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {
        "train": train_result[0]["ptl/test_MulticlassMatthewsCorrCoef"],
        "test": test_result[0]["ptl/test_MulticlassMatthewsCorrCoef"],
    }

    return model, result


# %% AND NOW TRAIN
print(" # Training Logistic regression model...")
results = {}
ratio_full_dataset = [0.25, 0.5, 1.0]
experiments_num_imgs = [
    int(np.floor(len(train_feats_simclr) * r)) for r in ratio_full_dataset
]

for num_imgs in experiments_num_imgs:
    sub_train_set = get_smaller_dataset(train_feats_simclr, num_imgs)
    _, small_set_results = train_logreg(
        batch_size=64,
        max_epochs=500,
        train_feats_data=sub_train_set,
        test_feats_data=test_feats_simclr,
        model_suffix=num_imgs,
        feature_dim=train_feats_simclr.tensors[0].shape[1],
        num_classes=3,
        lr=1e-3,
        weight_decay=1e-3,
    )
    results[num_imgs] = small_set_results

dataset_sizes = sorted([k for k in results])
test_scores = [results[k]["test"] for k in dataset_sizes]
print(" # Done!")
# %% PLOT RESULTS
fig = plt.figure(figsize=(6, 4))
plt.plot(
    dataset_sizes,
    test_scores,
    "--",
    color="#000",
    marker="*",
    markeredgecolor="#000",
    markerfacecolor="y",
    markersize=16,
)
plt.xscale("log")
plt.xticks(dataset_sizes, labels=dataset_sizes)
plt.title("CBTN classification over dataset size", fontsize=14)
plt.xlabel("Number of images")
plt.ylabel("Validation MCC")
plt.minorticks_off()
plt.show()
fig.savefig(os.path.join(CHECKPOINT_PATH, "CBTN_results.png"))
# plt.close()
for k, score in zip(dataset_sizes, test_scores):
    print(f"Test MCC for {k:3d} images per label: {score:4.2f}%")
