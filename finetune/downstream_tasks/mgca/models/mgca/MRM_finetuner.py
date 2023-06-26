import datetime
import os
from argparse import ArgumentParser

import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
import torchvision
from mgca.datasets.classification_dataset import (CheXpertImageDataset,
                                                  COVIDXImageDataset,
                                                  RSNAImageDataset)
from mgca.datasets.data_module import DataModule
from mgca.datasets.transforms import DataTransforms, Moco2Transform
from mgca.models.mgca.mgca_module import MGCA
from mgca.models.ssl_finetuner import SSLFineTuner
import os.path as osp
import os 
import logging
from model_MedKLIP import MedKLIP
import torch
import torch.nn as nn
from functools import partial
import timm
import os
import os.path as osp
from timm.models.vision_transformer import VisionTransformer


import warnings
warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="chexpert")
    parser.add_argument("--path", type=str,
                        default="None")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_pct", type=float, default=0.01)
    parser.add_argument("--model_name", type=str, default="MRM")
    parser.add_argument("--save_path_name", type=str, default="MRM")

    
    # add trainer args
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # set max epochs
    args.max_epochs = 50
    
    ############################# set the path #####################
    
    
    if args.model_name == 'MRM':
        print('loading MRM model')
        args.path = osp.join(os.path.abspath('.'), 'baseline_models/MRM/MRM.pth')
    elif args.model_name == 'mmodal_6':
        print('loading mmodal_6')
        args.path = osp.join(os.path.abspath('.'), "pretrain_models/pretrain_mmodal_new_6/epoch_80/cross-lingual_multi-modal_encoder.pth")
    elif args.model_name == 'mmodal_12':
        print('loading mmodal_12')
        
        args.path = osp.join(os.path.abspath('.'), "pretrain_models/pretrain_mmodal_new_12/epoch_60/cross-lingual_multi-modal_encoder.pth")
    elif args.model_name == 'resnet50':
        args.path = osp.join(os.path.abspath('.'), "pretrain_models/resnet50_imagnet/resnet50imageNet.pth")
    elif args.model_name == 'MGCA':
        print('loading MGCA')
        args.path = osp.join(os.path.abspath('.'), "pretrain_models/mgca_model/mgca_resnet_50.ckpt")
        

    seed_everything(args.seed)

    if args.dataset == "chexpert":
        # define datamodule
        # check transform here
        datamodule = DataModule(CheXpertImageDataset, None,
                                Moco2Transform, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 5
        multilabel = True
    elif args.dataset == "rsna":
        datamodule = DataModule(RSNAImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 1
        multilabel = True
    elif args.dataset == "covidx":
        datamodule = DataModule(COVIDXImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 2
        multilabel = False
    else:
        raise RuntimeError(f"no dataset called {args.dataset}")

    class resnet_bb(torch.nn.Module):
        def __init__(self, model, ckpt=None):
            super().__init__()
            self.backbone = model
            self.backbone.fc = torch.nn.Identity()
            self.backbone.load_state_dict(torch.load(ckpt, map_location='cpu'))
        def forward(self, x):
            return self.backbone(x)
    
    class ViT_encoder(nn.Module):
        def __init__(self, model, ckpt=None):
            super().__init__()
            self.vit_base = model
            
        def vit_forward(self, x):
            return self.vit_base(x)
        
        def forward(self, x):
            img_feat = self.vit_forward(x)
#             print(img_feat.contiguous().shape)
            return img_feat.unsqueeze(1).contiguous()
        
    def vit_base_patch16(**kwargs):
        model = VisionTransformer(norm_layer=partial(nn.LayerNorm, eps=1e-6),**kwargs)
        return model


    if args.path :
        
        if args.model_name == 'resnet50':
            
            model = torchvision.models.resnet50(pretrained=False)
            args.backbone = resnet_bb(model, args.path).backbone
            print('load ckpt', args.path)
            
        elif args.model_name == 'mmodal_3':
            model = torchvision.models.resnet50(pretrained=False)
            args.backbone = resnet_bb(model, args.path).backbone
            print('load ckpt', args.path)
        elif args.model_name == 'mmodal_6':
            model = torchvision.models.resnet50(pretrained=False)
            args.backbone = resnet_bb(model, args.path).backbone
            print('load ckpt', args.path)
        elif args.model_name == 'mmodal_12':
            model = torchvision.models.resnet50(pretrained=False)
            args.backbone = resnet_bb(model, args.path).backbone
            print('load ckpt', args.path)
            
        elif args.model_name == 'MGCA':
            model = torchvision.models.resnet50(pretrained=False)
            args.backbone = resnet_bb(model, args.path).backbone
            print('load ckpt', args.path)
            
        elif args.model_name == 'MRM':
            
            # model definition
            model = vit_base_patch16(num_classes=14, drop_path_rate=0.1)
            ckpt_path = osp.join(os.path.abspath('.'), 'baseline_models/MRM/MRM.pth')
            checkpoint_model = torch.load(ckpt_path, map_location="cpu")["model"]
            # load the pre-trained model
            model.load_state_dict(checkpoint_model, strict=False)
            args.backbone = ViT_encoder(model)
        
    else:
        model = torchvision.models.resnet50(pretrained=True)

    args.num_classes = num_classes
    args.multilabel = multilabel

    # finetune


    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/mgca_finetune/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    if args.dataset == "rsna" or args.dataset == "chexpert":
        
    
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(monitor="val_auc", dirpath=ckpt_dir,
                            save_last=True, mode="max", save_top_k=1),
            EarlyStopping(monitor="val_auc", min_delta=0.,
                          patience=5, verbose=False, mode="max")
        ]
    else:
        
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(monitor="val_acc", dirpath=ckpt_dir,
                            save_last=True, mode="max", save_top_k=1),
            EarlyStopping(monitor="val_acc", min_delta=0.,
                          patience=15, verbose=False, mode="max")
        ]
        

    # get current time
    now = datetime.datetime.now(tz.tzlocal())

    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    
    ############################# save result path ########################################################
    result_path = osp.join(os.path.abspath('.'), 'logs/' + f"{args.model_name}_{args.dataset}_{args.data_pct}_{extension}")
    
    logger_dir = os.path.join(
        BASE_DIR, result_path)
    
    os.makedirs(logger_dir, exist_ok=True)
    ######################################################################################################
    
    
    wandb_logger = WandbLogger(
        project="mgca_finetune",
        save_dir=logger_dir,
        name=f"{args.model_name}_{args.dataset}_{args.data_pct}_{extension}")
    trainer = Trainer.from_argparse_args(
        args,
        deterministic=True,
        callbacks=callbacks,
        logger=wandb_logger)

    
    tuner = SSLFineTuner(**args.__dict__)
    
    tuner.training_steps = tuner.num_training_steps(trainer, datamodule)
    

    # train
    trainer.fit(tuner, datamodule)
    # test
    trainer.test(tuner, datamodule, ckpt_path="best")


if __name__ == "__main__":
    cli_main()
