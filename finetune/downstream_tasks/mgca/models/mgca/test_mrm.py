import torch
import torch.nn as nn
from functools import partial
import timm
import os
import os.path as osp
from timm.models.vision_transformer import VisionTransformer

def vit_base_patch16(**kwargs):
    model = VisionTransformer(norm_layer=partial(nn.LayerNorm, eps=1e-6),**kwargs)
    return model

# model definition
model = vit_base_patch16(num_classes=14,drop_path_rate=0.1)
ckpt_path = osp.join(os.path.abspath('.'), 'baseline_models/MRM/MRM.pth')
checkpoint_model = torch.load(ckpt_path, map_location="cpu")["model"]
# load the pre-trained model
model.load_state_dict(checkpoint_model, strict=False)