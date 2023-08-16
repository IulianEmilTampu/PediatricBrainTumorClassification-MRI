# %% IMPORTS
import os
import pathlib
import glob
import matplotlib.pyplot as plt
import numpy as np
import PIL

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models
import torchvision.transforms as T
from strong_augment import StrongAugment

import pytorch_lightning as pl
import torchmetrics
import seaborn as sns
import pandas as pd

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


# CODE FROM https://docs.ray.io/en/latest/train/examples/pytorch/pytorch_resnet_finetune.html
# %% define data transofms
if __name__ == '__main__':
    ImageNet_mean = np.array([0.4451, 0.4262, 0.3959])  # mean of your dataset
    ImageNet_std = np.array([0.2411, 0.2403, 0.2466]) # std of your dataset

    train_trnsf = T.Compose([
        T.RandomRotation(20),
        T.RandomEqualize(0.2),
        T.RandomPosterize(8,p=0.2),
        T.RandomResizedCrop(size=(224,224), scale=(0.5,1.0), ratio=(0.85, 1.5), antialias=True),
        T.RandomVerticalFlip(0.5),
        T.RandomHorizontalFlip(0.5),
        # StrongAugment(), # Just one line!
        T.ToTensor(),
        T.Normalize(mean=ImageNet_mean, std=ImageNet_std)],
    )

    val_trnsf = T.Compose([
        T.Resize(size=(224,224), antialias = True),
        T.ToTensor(),
        T.Normalize(mean=ImageNet_mean, std=ImageNet_std)],
    )


    # %% CREATE DATASET

    print('Creating datagenerators...')
    BATCH_SIZE = 64

    TRAINING_DATASET_PATH =pathlib.Path( "C:\\Datasets\\CBTN_v1\\TEST_JPG_DATASET_VERSION\\TRAINING").with_suffix('')
    VALIDATION_DATASET_PATH =pathlib.Path( "C:\\Datasets\\CBTN_v1\\TEST_JPG_DATASET_VERSION\\VALIDATION").with_suffix('')

    torch_dataset = {
            'train': datasets.ImageFolder(TRAINING_DATASET_PATH, transform=train_trnsf),
            'val' : datasets.ImageFolder(VALIDATION_DATASET_PATH, transform=val_trnsf),
            'test' : datasets.ImageFolder(VALIDATION_DATASET_PATH, transform=val_trnsf),
        }

    print('Training...')
    train_data = DataLoader(torch_dataset['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    print('Done!')
    print('Validation...')
    valid_data = DataLoader(torch_dataset['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    print('Done!')
    print('Testing...')
    test_data = DataLoader(torch_dataset['test'], batch_size=BATCH_SIZE, shuffle=False)
    print('Done!')

    # plot some example images
    plot_examples = False
    if plot_examples:
        train_features, train_labels = next(iter(train_data))
        figure = plt.figure(figsize=(15,40))
        for i in range(1,6):
            img = train_features[i-1].permute(1,2,0)
            label = train_labels[i-1]
            figure.add_subplot(16,5,i)
            plt.title(label)
            plt.axis('off')
            plt.imshow(ImageNet_std * img.numpy() + ImageNet_mean)
    # %% DEFINE MODEL (USE PYTORCH LIGHNING)
    # https://www.kaggle.com/code/shreydan/resnet50-pytorch-lightning-kfolds

    class ResNet50Model(pl.LightningModule):
        
        def __init__(self, pretrained=True, in_channels = 3, num_classes = 3, lr=3e-4, freeze=True):
            super(ResNet50Model, self).__init__()
            self.in_channels = in_channels
            self.num_classes = num_classes
            self.lr = lr
            
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
            
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, 128),
                nn.Dropout(0.3),
                nn.Linear(128, self.num_classes)
            )
            
            self.loss_fn = nn.CrossEntropyLoss()
            
            self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
            self.train_MCC = torchmetrics.MatthewsCorrCoef(task='multiclass', num_classes=num_classes)
            self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
            self.val_MCC = torchmetrics.MatthewsCorrCoef(task='multiclass', num_classes=num_classes)
            self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
            self.test_MCC = torchmetrics.MatthewsCorrCoef(task='multiclass', num_classes=num_classes)

            # for teh confusion matrix
            self.test_step_y_hats = []
            self.test_step_ys = []

            # for saving some training, validation, testing images
            self.validation_step_img = []
            self.training_step_img = []
            self.test_step_img = []

            # mean and std for rescaling image input
            self.ImageNet_mean = torch.Tensor([0.4451, 0.4262, 0.3959]).unsqueeze(-1).unsqueeze(-1)
            self.ImageNet_std = torch.Tensor([0.2411, 0.2403, 0.2466]).unsqueeze(-1).unsqueeze(-1)



        def forward(self, x):
            return self.model(x)
        
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
            return [optimizer], [scheduler]
        
        def training_step(self, batch, batch_idx):
            
            x, y = batch
            
            preds = self.model(x)
            
            loss = self.loss_fn(preds, y)

            # compute metrics
            self.train_acc.update(torch.argmax(preds, dim=1), y)
            self.train_MCC.update(torch.argmax(preds, dim=1), y)
            
            self.log('train_loss', loss.item(), on_epoch=True)
            self.log('train_acc', self.train_acc, on_epoch=True)
            self.log('train_MCC', self.train_MCC, on_epoch=True)

            # save one batch of images
            if len(self.training_step_img) == 0:
                self.training_step_img.append(x.to('cpu') * self.ImageNet_std + self.ImageNet_mean)
            
            return loss
        
        def validation_step(self, batch, batch_idx):
            
            x,y = batch
            
            preds = self.model(x)
            
            loss = self.loss_fn(preds, y)

            # compute metrics
            self.val_acc.update(torch.argmax(preds, dim=1), y)
            self.val_MCC.update(torch.argmax(preds, dim=1), y)

            self.log('val_loss', loss.item(), on_epoch=True)
            self.log('val_acc', self.val_acc, on_epoch=True)
            self.log('val_MCC', self.val_MCC, on_epoch=True)

            # save images for tensorboard saving
            # save one batch of images
            if len(self.validation_step_img) == 0:
                self.validation_step_img.append(x.to('cpu') * self.ImageNet_std + self.ImageNet_mean)
            
        def test_step(self, batch, batch_idx):

            x,y = batch
            preds = self.model(x)

            # compute metrics
            self.test_acc.update(torch.argmax(preds, dim=1), y)
            self.test_MCC.update(torch.argmax(preds, dim=1), y)
            
            self.log('test_acc', self.test_acc, on_epoch=True)
            self.log('test_MCC', self.test_MCC, on_epoch=True)

            # save for confusion matrix
            self.test_step_y_hats.append(preds)
            self.test_step_ys.append(y)

            # save images for tensorboard saving
            # save one batch of images
            if len(self.test_step_img) == 0:
                self.test_step_img.append(x.to('cpu') * self.ImageNet_std + self.ImageNet_mean)
        
        def on_train_epoch_end(self) -> None:
            _imgs = torch.cat(self.training_step_img)
            img_grid = torchvision.utils.make_grid(_imgs).cpu()
            self.loggers[0].experiment.add_image("Example_train_batch", img_grid, self.current_epoch)
        
        def on_validation_epoch_end(self) -> None:
            _imgs = torch.cat(self.validation_step_img)
            img_grid = torchvision.utils.make_grid(_imgs).cpu()
            self.loggers[0].experiment.add_image("Example_validation_batch", img_grid, self.current_epoch)

        def on_test_epoch_end(self):

            # save confusion matrix
            y_hat = torch.cat(self.test_step_y_hats).cpu()
            y = torch.cat(self.test_step_ys).cpu()

            confusion_matrix = torchmetrics.ConfusionMatrix(task = 'multiclass', num_classes=self.num_classes, threshold=0.05)
            confusion_matrix(torch.argmax(y_hat, dim=1), y.int())

            confusion_matrix_computed = confusion_matrix.compute().detach().cpu().numpy().astype(int)

            df_cm = pd.DataFrame(confusion_matrix_computed)
            plt.figure(figsize = (10,7))
            fig_ = sns.heatmap(df_cm, annot=True, cmap='Blues', annot_kws={"size": 20}).get_figure()
            plt.close(fig_)
            self.loggers[0].experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

            # save test images
            _imgs = torch.cat(self.test_step_img)
            img_grid = torchvision.utils.make_grid(_imgs).cpu()
            self.loggers[0].experiment.add_image("Example_test_batch", img_grid, self.current_epoch)

            # (TODO) plot histogram of a bunch of train, validation and test images
            # test images
            # hist_data = torch.cat(self.training_step_img).numpy()[0:10]
            # df = pd.DataFrame(hist_data.reshape((3,-1)).T)
            # df["id"] = df.index
            # df = pd.melt(df, id_vars='id', value_vars=[0, 1, 2])
            # plt.figure(figsize = (10,7))
            # fig_ = sns.histplot(data=df, x="value", hue="variable")
            # self.loggers[0].experiment.add_image("Historgam_train_images", sns.histplot(data=df, x="value", hue="variable"), self.current_epoch)

    # %% TRAIN MODEL

    lr = 0.001
    logs = dict()
    fold = 1

    model = ResNet50Model()
    trainer = pl.Trainer(accelerator='gpu', 
                            max_epochs=50,
                            callbacks=[
                            EarlyStopping(monitor="val_loss", 
                                        mode="min",
                                        patience=50,
                                        )
                            ],
                            logger=TensorBoardLogger(os.path.join(os.getcwd(), 'trained_model_archive', 'TEST_py_RESNET250'), name="my_model"),
                            log_every_n_steps=40,
                        )

    model.hparams.lr = lr

    trainer.fit(model, train_data, valid_data)
    metrics = trainer.logged_metrics
    trainer.test(model, test_data)

    logs[f'fold{fold}'] = {
        'train_loss': metrics['train_loss_epoch'].item(),
        'val_loss': metrics['val_loss'].item(),
        'train_acc': metrics['train_acc_epoch'].item(),
        'val_acc': metrics['val_acc'].item(),
        'train_MCC': metrics['train_MCC_epoch'].item(),
        'val_MCC': metrics['val_MCC'].item(),

    }

    print(f"Train Loss: {logs[f'fold{fold}']['train_loss']} | Train Accuracy: {logs[f'fold{fold}']['train_acc']} | Train MCC: {logs[f'fold{fold}']['train_MCC']}")
    print(f"Val Loss: {logs[f'fold{fold}']['val_loss']} | Val Accuracy: {logs[f'fold{fold}']['val_acc']} | Val MCC: {logs[f'fold{fold}']['val_MCC']}")
