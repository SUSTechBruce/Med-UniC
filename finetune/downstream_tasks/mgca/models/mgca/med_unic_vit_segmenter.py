import sys
sys.settrace
import datetime
import os
from argparse import ArgumentParser

import segmentation_models_pytorch as smp
import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger

from mgca.datasets.data_module import DataModule
from mgca.datasets.segmentation_dataset import (RSNASegmentDataset,
                                                SIIMImageDataset)
from mgca.models.backbones.transformer_seg import SETRModel
from mgca.models.mgca.mgca_module import MGCA
from mgca.models.ssl_segmenter import SSLSegmenter
import os.path as osp
import os 
from vits import create_vit
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def cli_main():
    parser = ArgumentParser(
        "Finetuning of semantic segmentation task for MGCA")
    parser.add_argument("--base_model", type=str,
                        default="vit", help="resnet50 or vit")
    parser.add_argument("--ckpt_path", type=str,
                        default="/home/cheliu/phd_work/ckpt/mgca_resnet_50.ckpt")
    parser.add_argument("--dataset", type=str, default="siim")
    parser.add_argument("--seed", type=int, default=15) # 15
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument("--data_pct", type=float, default=0.01)
    parser.add_argument("--model_name", type=str,
                        default="mmodal_3", help="resnet50 or vit")
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
# CUDA_VISIBLE_DEVICES=0 python mgca_segmenter.py --gpus 1 --data_pct 1 --dataset siim --learning_rate 5e-4 --epochs 100 --batch_size 4 --seed 42
    # args.deterministic = True
    args.max_epochs = args.epochs

    seed_everything(args.seed)

    if args.dataset == "siim":
        datamodule = DataModule(SIIMImageDataset, None,
                                None, args.data_pct,
                                args.batch_size, args.num_workers)
    elif args.dataset == "rsna":
        datamodule = DataModule(RSNASegmentDataset, None,
                                None, args.data_pct,
                                args.batch_size, args.num_workers)
        
    if args.model_name == 'mmodal_3':
        
        args.ckpt_path = osp.join(os.path.abspath('.'), "pretrain_models/freeze_12_test/epoch55/cross-lingual_multi-modal_encoder.pth")
        print('Loading from args.ckpt_path: ', args.ckpt_path)
        
    elif args.model_name == 'mmodal_6':
        print('loading mmodal_6 is ok')
        args.ckpt_path = osp.join(os.path.abspath('.'), "pretrain_models/pretrain_mmodal_new_6/epoch_100/cross-lingual_multi-modal_encoder.pth")
    elif args.model_name == 'mmodal_12':
        print('loading mmodal_12 is ok')
        args.ckpt_path = osp.join(os.path.abspath('.'), "pretrain_models/pretrain_mmodal_new_12/epoch_80/cross-lingual_multi-modal_encoder.pth")
        
    elif args.model_name == 'resnet50':
        args.ckpt_path = osp.join(os.path.abspath('.'), "pretrain_models/resnet50_imagnet/resnet50imageNet.pth")
    elif args.model_name == 'MGCA':
        print('loading MGCA')
        args.ckpt_path = osp.join(os.path.abspath('.'), "pretrain_models/mgca_model/mgca_resnet_50.ckpt")
    

    # mgca = MGCA.load_from_checkpoint(args.ckpt_path)
    # encoder = mgca.img_encoder_q.model

    if args.base_model == "vit":
        args.seg_model = SETRModel(
            patch_size=(16, 16),
            in_channels=3,
            out_channels=1,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            decode_features=[512, 256, 128, 64]
        )
        
        # model definition
        vit_grad_ckpt = False
        vit_ckpt_layer = 0
        image_size = 512
        vit_name = 'base'
        model, vision_with = create_vit(vit_name, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
        ckpt_path = osp.join(os.path.abspath('.'), 'pretrain_models/vit_base/epoch60/cross-lingual_multi-modal_total.pth')
        print(ckpt_path)
        checkpoint_model = torch.load(ckpt_path, map_location="cpu")
        # load the pre-trained model
        model.load_state_dict(checkpoint_model, strict=False)
        args.seg_model.encoder_2d.bert_model = model

        for param in args.seg_model.encoder_2d.bert_model.parameters():
            param.requires_grad = False

    elif args.base_model == "resnet50":
        # FIXME: fix this later
        args.seg_model = smp.Unet(
            args.base_model, encoder_weights=None, activation=None)

        if args.ckpt_path:
            ckpt = torch.load(args.ckpt_path)
            ckpt_dict = dict()

            args.seg_model.encoder.load_state_dict(ckpt)
            # Freeze encoder
            if args.dataset == 'rsna':
                for param in args.seg_model.encoder.parameters():
                    param.requires_grad = False
            else:
                for param in args.seg_model.encoder.parameters():
                    param.requires_grad = False
    model = SSLSegmenter(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/segmentation/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_dice", dirpath=ckpt_dir,
                        save_last=True, mode="max", save_top_k=1),
        EarlyStopping(monitor="val_dice", min_delta=0.,
                      patience=20, verbose=False, mode="max")
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"../../../data")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="segmentation", save_dir=logger_dir,
        name=f"MGCA_{args.dataset}_{args.data_pct}_{extension}")
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=1)

    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps, '!!!!')
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path='best')


if __name__ == "__main__":
    cli_main()
