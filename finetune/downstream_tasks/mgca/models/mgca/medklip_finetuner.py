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
import ruamel_yaml as yaml
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
import warnings
import torch.nn as nn
warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


#     (5): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)

def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="chexpert")
    parser.add_argument("--path", type=str,
                        default="None")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_pct", type=float, default=0.01)
    parser.add_argument("--model_name", type=str, default="medklip")
    parser.add_argument("--save_path_name", type=str, default="medklip")

    
    # add trainer args
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # set max epochs
    args.max_epochs = 50
    
    ############################# set the path #####################
    if args.model_name == 'medklip':
        
        print('loading medklip')
        args.path = osp.join(os.path.abspath('.'), "baseline_models/MedKLIP/checkpoint_final.pth")
        
    elif args.model_name == 'resnet50':
        args.path = osp.join(os.path.abspath('.'), "pretrain_models/resnet50_imagnet/resnet50imageNet.pth")
        
    
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
            self.backbone.load_state_dict(torch.load(ckpt, map_location='cpu'))
            self.backbone.fc = torch.nn.Identity()

        def forward(self, x):
            return self.backbone(x)
        
    class resnet_medklip(torch.nn.Module):
        def __init__(self, model, ckpt=None):
            super().__init__()
            self.backbone = nn.Sequential(
                model.res_features,
                nn.Linear(224*224, 2048)    
            )
            

        def forward(self, x):
            return self.backbone(x)
        
    if args.path :
        
        if args.model_name == 'resnet50':
            
            model = torchvision.models.resnet50(pretrained=False)
            args.backbone = resnet_bb(model, args.path).backbone
            print('load ckpt', args.path)
            print(args.backbone)

            
        elif args.model_name == 'medklip':
            config = osp.join(os.path.abspath('.'), "Pretrain_MedKLIP.yaml")
            config = yaml.load(open(config, 'r'), Loader=yaml.Loader)

            print("Creating model")
            model = MedKLIP(config,None, None, mode = 'train')
            device = 'cuda'
            model = model.to(device)
            print('Start to load model')
            checkpoint_path =  osp.join(os.path.abspath('.'), "baseline_models/MedKLIP/checkpoint_final.pth")                             
            checkpoint = torch.load(checkpoint_path, map_location='cpu') 
            state_dict = checkpoint['model']


            weights_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('module.', '') if 'module' in k else k
                weights_dict[new_k] = v


            model.load_state_dict(weights_dict)
            
            args.backbone = model.res_features
            
            print(args.backbone)

            print('loading ok')
            print('load ckpt', args.path)
        
    else:
        print('Something wrong')
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
