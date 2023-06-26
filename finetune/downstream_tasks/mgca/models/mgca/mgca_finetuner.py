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
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_pct", type=float, default=0.01)
    parser.add_argument("--model_name", type=str, default="mmodal_3")
    parser.add_argument("--save_path_name", type=str, default="mmodal_3")

    
    # add trainer args CUDA_VISIBLE_DEVICES=1 python mgca_finetuner.py --gpus 1 --dataset covidx --data_pct 1 --batch_size 4 --seed 4550
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # set max epochs
    args.max_epochs = 50
    
    ############################# set the path #####################
    if args.model_name == 'mmodal_3':
        print('loading mmodal_3')
        args.path = osp.join(os.path.abspath('.'), "pretrain_models/ablation_512/cross-lingual_multi-modal_encoder.pth")
    elif args.model_name == 'pretrain_unified_3':
        print('loading pretrain_unified_3')
        args.path = osp.join(os.path.abspath('.'), "pretrain_models/pretrain_unified_3/epoch_70/cross-lingual_multi-modal_encoder.pth")
    elif args.model_name == 'from_scratch':
        
        args.path = osp.join(os.path.abspath('.'), "pretrain_models/from_scratch_3/epoch60/cross-lingual_multi-modal_encoder.pth")
        
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
        

    if args.path :
        
        if args.model_name == 'resnet50':
            
            model = torchvision.models.resnet50(pretrained=False)
            args.backbone = resnet_bb(model, args.path).backbone
            print('load ckpt', args.path)
            
        elif args.model_name == 'mmodal_3':
            model = torchvision.models.resnet50(pretrained=False)
            args.backbone = resnet_bb(model, args.path).backbone
            print('load ckpt', args.path)
        elif args.model_name == 'pretrain_unified_3':
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
