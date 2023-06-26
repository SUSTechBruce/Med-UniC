import utils_builder
import utils_dataset

from utils_optimizer import LARS
import torch
import numpy as np

# import wandb
import os

import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
from apex.parallel import DistributedDataParallel as DDP
import random
from utils_trainer import trainer_wBert
from argparse import ArgumentParser
import logging

oncloud = True
try:
    import moxing as mox
except:
    oncloud = False

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def ddp_main():
    parser = ArgumentParser()

    # input data
    parser.add_argument('--train_target_csv_path', type=str,
                        default='/raid/cl522/MIMIC-CXR/Image-Text/9W_MIMIC_train.csv')
    parser.add_argument('--en_img_path', type=str, required=True,
                        default='/raid/cl522/MIMIC-CXR/Image-Text/9W_MIMIC_train.npy')
    parser.add_argument('--en_text_csv_path', type=str, required=True,
                        default='/raid/cl522/MIMIC-CXR/Image-Text/MIMIC-CXR-meta_INDI_FIND_IMP_report.csv')
    parser.add_argument('--sp_img_path', type=str, required=True, default='/raid/cl522/MIMIC-CXR/Image-Text/PDC.npy')
    parser.add_argument('--sp_text_csv_path', type=str, required=True,
                        default='/raid/cl522/MIMIC-CXR/Image-Text/new_pdc.csv')

    # load model
    parser.add_argument('--model', type=str, required=True, default='/cxr_bert')
    parser.add_argument('--un_pretrain_model', type=str, required=True, default='/cxr_bert')

    # trainer
    parser.add_argument('--batch_size', type=int, required=True, default=128)
    parser.add_argument('--test_batch_size', type=int, default=200)
    parser.add_argument('--checkpoint_interval', type=int, default=100000)
    parser.add_argument('--lr', type=float, required=True, default=5.0e-3)
    parser.add_argument('--max_epochs', type=int, required=True, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--test_interval', type=int, default=2)
    parser.add_argument('--loss_type', type=str, default='unified_loss')
    parser.add_argument('--smooth', type=str, default='exp')
    parser.add_argument('--ratio', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=1.0e-10)
    parser.add_argument('--logging_steps', type=int, default=5)
    parser.add_argument("--max_seq_length", required=True, type=int, default=512)
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")

    # network
    parser.add_argument('--freeze_layers', type=int, required=True, default=12)
    parser.add_argument('--feature_dim', type=int, default=768)
    parser.add_argument('--mlp_hidden_size', type=int, default=2048)
    parser.add_argument('--projection_size', type=int, default=768)

    # default pretraining params
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--nas_output_dir', type=str, required=True,
                        default='s3://bucket-375/hebin/code/zen/mwe/output')
    parser.add_argument('--gradient_accumulation_steps', type=int, required=True, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    # mdoel name
    parser.add_argument('--img_model', type=str, default='resnet50')
    parser.add_argument('--text_model', type=str, default='bert')
    parser.add_argument('--text_model_arch', type=str, default='general')
    parser.add_argument('--model_name', type=str, default='cross-lingual_multi-modal')
    parser.add_argument('--cache_dir', type=str, required=True, default=None, help='')
    parser.add_argument('--vision_model_path', type=str, required=True, default=None, help='')
    parser.add_argument('--lambda_t', type=float, required=True, default=0.1)
    parser.add_argument('--text_aug', type=int, required=True, default=0)
    parser.add_argument('--from_scratch', type=int, required=True, default=0)

    # vit encoder:
    parser.add_argument('--vision_encoder_name', type=str, required=True, default='vit')
    parser.add_argument('--vit_path', type=str, required=True, default='/vit')
    parser.add_argument('--vit_name', type=str, required=True, default='base')

    # apex mix training
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )

    args = parser.parse_args()

    assert (torch.cuda.is_available())
    device_count = torch.cuda.device_count()
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    init_method = ''
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port

    torch.cuda.set_device(args.local_rank)
    device_id = torch.device("cuda", args.local_rank)
    print('device_id: %s' % args.local_rank)
    print('device_count: %s, rank: %s, world_size: %s' % (device_count, args.rank, args.world_size))
    print(init_method)

    args.device = device_id

    torch.distributed.init_process_group(backend='nccl', world_size=args.world_size,
                                         rank=args.rank, init_method=init_method)
    torch.cuda.empty_cache()

    # setting the default store path
    LOCAL_DIR = args.nas_output_dir
    if args.rank == 0:
        if not os.path.exists(LOCAL_DIR):
            os.makedirs(LOCAL_DIR)
            logger.info(LOCAL_DIR + ' created!')

    # list model path
    if args.local_rank == 0:
        print('Moxing successfully #################')

        logging.info(mox.file.list_directory(args.model, recursive=True))

    save_name = '_'.join([
        '{}'.format(args.model_name),
        'epoch', str(args.max_epochs),
        'lr', str(args.lr),
        'bsz', str(args.batch_size),
        'grad_accu', str(args.gradient_accumulation_steps),
        str(args.max_seq_length),
        'gpu', str(args.world_size),
    ])

    local_save_dir = os.path.join(LOCAL_DIR, 'output', 'multimodal', 'checkpoints')
    tensor_dir = os.path.join(LOCAL_DIR, 'output', 'multimodal', 'tensorboard')

    bash_save_dir = os.path.join(local_save_dir, save_name)
    bash_tsbd_dir = os.path.join(tensor_dir, save_name)

    if args.rank == 0:
        if not os.path.exists(bash_save_dir):
            os.makedirs(bash_save_dir)
            logger.info(bash_save_dir + ' created!')

        if not os.path.exists(bash_tsbd_dir):
            os.makedirs(bash_tsbd_dir)
            logger.info(bash_tsbd_dir + ' created!')

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.batch_size = args.batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # loading data path
    en_text_path = args.en_text_csv_path  # need to parser
    en_img_path = args.en_img_path
    sp_text_path = args.sp_text_csv_path
    sp_img_path = args.sp_img_path

    img_path = {'en_img_path': en_img_path, 'sp_img_path': sp_img_path}
    csv_path = {'en_text_path': en_text_path, 'sp_text_path': sp_text_path}

    # define image-text dataset
    # train_dataset = utils_dataset.I_T_emb_dataset(image_path=img_path, csv_path=text_path, **text_emb_path)
    train_dataset = utils_dataset.I_T_emb_dataset(image_path=img_path, csv_path=csv_path)
    train_dataset = train_dataset.get_dataset(train_test='train')

    # building model part
    # --------------------

    model = utils_builder.ResNet_CXRBert(args)

    model.to(device_id)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    trainer = trainer_wBert(model=model,
                            optimizer=optimizer,
                            device=device_id,
                            model_name='cross-lingual_multi-modal',
                            args=args)

    trainer.train_w_TextEmb(train_dataset, bash_save_dir, bash_tsbd_dir)


ddp_main()