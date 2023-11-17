# %%
import os
import torch
from torchvision import models
from torchsummary import summary
from pytorch_lightning.callbacks import ModelSummary

import model_bucket_CBTN_v1
import importlib

# %%
importlib.reload(model_bucket_CBTN_v1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
model = model_bucket_CBTN_v1.ResNets(
    nbr_classes=3, version="resnet9", pretrained=True, freeze_percentage=0.6
)


child_counter = 0
for child in model.children():
    print(" child", child_counter, "is -")
    print(child)
    child_counter += 1

summary(model.to(device), (3, 240, 240))
# %%
importlib.reload(model_bucket_CBTN_v1)

model = model_bucket_CBTN_v1.ResNetModel(
    version="resnet50",
    pretrained=True,
    freeze_percentage=0.7,
    class_weights=(1, 1, 1),
)

# %%
import monai
import numpy as np

net = monai.networks.nets.ViT(num_classes=3, patch_size=16, in_channels=3, img_size=(224,224), proj_type='conv', pos_embed_type='sincos', classification=True, spatial_dims=2,)

# %%
import numpy as np
import torch
test_input = torch.tensor(np.ones((2, 3, 224, 224)).astype('float32'))

with torch.no_grad():
    out = net(test_input)

# %%
import torch 
import torchvision

net = torchvision.models.vit_b_16()
