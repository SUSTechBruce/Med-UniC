# -*- coding:utf-8 -*-

import moxing as mox
import os
import argparse
import logging
from pathlib import Path

os.environ["NCCL_DEBUG"] = "INFO"

parser = argparse.ArgumentParser()
## default roma parameters
parser.add_argument('--init_method', type=str, default=None, help='cloud will handle this, do not set any value by yourself')
parser.add_argument('--rank', type=int, default=0, help='cloud will handle this, do not set any value by yourself')
parser.add_argument('--world_size', type=int, default=1, help='cloud will handle this, do not set any value by yourself')
parser.add_argument('--nproc_per_node', type=int, default=0, help='cloud will handle this, do not set any value by yourself')
parser.add_argument('--nnodes', type=int, default=1, help='cloud will handle this, do not set any value by yourself')

## input multi-modal data
parser.add_argument('--en_img_path', type=str, required=True,
                    default='/raid/cl522/MIMIC-CXR/Image-Text/9W_MIMIC_train.npy')
parser.add_argument('--en_text_csv_path', type=str, required=True,
                    default='/raid/cl522/MIMIC-CXR/Image-Text/MIMIC-CXR-meta_INDI_FIND_IMP_report.csv')
parser.add_argument('--sp_img_path', type=str, required=True, default='/raid/cl522/MIMIC-CXR/Image-Text/PDC.npy')
parser.add_argument('--sp_text_csv_path', type=str, required=True,
                    default='/raid/cl522/MIMIC-CXR/Image-Text/new_pdc.csv')

## default output file
parser.add_argument('--nas_output_dir', type=str, required=True, default='s3://bucket-375/hebin/code/zen/mwe/output', help='')
parser.add_argument('--cache_dir', type=str, required=True, default='/cache', help='')


## input model file
parser.add_argument('--freeze_layers', type=int, required=True, default=12)

parser.add_argument("--max_epochs", type=int, default=100, required=True, help="Number of epochs to train for")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    required=True,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--batch_size",
                    default=16,
                    required=True,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--lr",
                    default=5.0e-3,
                    required=True,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--max_seq_length",required=True, type=int, default=512)
parser.add_argument('--apex_folder', type=str, default='s3://bucket-373/hebin/tools/apex/')
parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--model', type=str, required=True, help='The path to save the model')
parser.add_argument('--un_pretrain_model', type=str, required=True, default='/cxr_bert')


parser.add_argument("--model_type", default='bert', type=str)
parser.add_argument("--load_model", default=0, type=int)
parser.add_argument("--warmup_steps", default=10000, type=int)
parser.add_argument('--tensor_dir', type=str, default='s3://bucket-373/hebin/tools/apex/')
parser.add_argument('--img_data', type=str, default='s3://bucket-373/hebin/tools/apex/')
parser.add_argument('--text_data', type=str, default='s3://bucket-373/hebin/tools/apex/')
parser.add_argument('--vision_model_path', type=str, default='s3://bucket-373/hebin/tools/apex/')
parser.add_argument('--loss_type', type=str, required=True, default='clip_loss')
parser.add_argument('--lambda_t', type=float, required=True, default=1)
parser.add_argument('--text_aug', type=int, required=True, default=0)
parser.add_argument('--from_scratch', type=int, required=True, default=0)

# vit encoder:
parser.add_argument('--vision_encoder_name', type=str, required=True, default='vit')
parser.add_argument('--vit_path', type=str, required=True, default='/vit')
parser.add_argument('--vit_name', type=str, required=True, default='base')

# prefix_token_len
args, _ = parser.parse_known_args()


################################# copy generate_data to mox ##############################
DS_DIR_NAME = "src"
os.environ['DLS_LOCAL_CACHE_PATH'] = "/cache"

LOCAL_DIR = os.environ['DLS_LOCAL_CACHE_PATH']
assert mox.file.exists(LOCAL_DIR)
logging.info("local disk: " + LOCAL_DIR)

dir_name = os.path.dirname(os.path.abspath(__file__))

# # copy data to local /cache/lcqmc
# logging.info("copying data...")
# local_dir = os.path.join(LOCAL_DIR, DS_DIR_NAME)

# copy data to local /cache/lcqmc
logging.info("copying data...")
local_data_dir = os.path.join(LOCAL_DIR, DS_DIR_NAME)

logging.info("copying en sp img data to yundao ...")

logging.info(mox.file.list_directory(args.img_data, recursive=True))
mox.file.copy_parallel(args.img_data, local_data_dir)

logging.info("copying en sp text data to yundao ...")

logging.info(mox.file.list_directory(args.text_data, recursive=True))
mox.file.copy_parallel(args.text_data, local_data_dir)


# copy valid data to local /cache/lcqmc

print('############################# copy img data ##############################')
#############################################
args.en_img_path = Path(os.path.join(local_data_dir, 'only_imp.npy'))
print('en img file: ', args.en_img_path)
args.sp_img_path = Path(os.path.join(local_data_dir, 'PDC_train_int.npy'))
print('sp img file: ', args.sp_img_path)

args.en_text_csv_path = Path(os.path.join(local_data_dir, '200k_find_imp.csv'))
print('en text file: ', args.en_text_csv_path)
args.sp_text_csv_path = Path(os.path.join(local_data_dir, 'PDC_cleaned.csv'))
print('sp text file: ', args.sp_text_csv_path)


host, port = args.init_method[:-5], args.init_method[-4:]
print(host, port)

init_method = args.init_method
rank = str(args.rank)
world_size = str(args.world_size)
print(init_method, rank, world_size)

print('Copy apex start...')
# copy apex-master folder to local machine
mox.file.copy_parallel(args.apex_folder, '/cache/apex-master')
os.system('python -m pip install --upgrade --force pip ')
os.system('python -m pip install transformers==4.9.0 ')
os.system('python -m pip install scikit-learn ')
os.system('python -m pip install timm ')
os.system('python -m pip install fairscale ')

# os.system('pip install setuptools==33.1.1 ')

os.system('pip --default-timeout=100 install -v --no-cache-dir'
          ' --global-option="--cpp_ext" --global-option="--cuda_ext" /cache/apex-master')

try:
    import apex
    print('Import apex success...')
    import amp_C
    print('Import amp_C success...')
    import apex_C
    print('Import apex_C success...')
except Exception:
    print('Install Apex failure...')

dir_name = '/home/ma-user/modelarts/user-job-dir/simplified_code/'

os.system('pip install -r ' + dir_name + 'requirements.txt')
os.system('pip install sklearn')

# os.system('cd ' + dir_name)
# logging.info(mox.file.list_directory(dir_name, recursive=True))

script = os.path.join(dir_name, 'run_pretraining_mmodal.sh')

cmd = ['bash', script, host, port, rank, args.nnodes, args.nproc_per_node,
       args.en_img_path, args.en_text_csv_path, args.sp_img_path,
       args.sp_text_csv_path, args.model, args.batch_size, args.lr,
       args.max_epochs, args.max_seq_length, args.freeze_layers,
       args.nas_output_dir, args.gradient_accumulation_steps,
       args.cache_dir, args.vision_model_path, args.loss_type, args.lambda_t, args.text_aug, args.un_pretrain_model, args.from_scratch,
       args.vision_encoder_name, args.vit_path, args.vit_name
       ]

bash_cmd = ' '.join([str(item) for item in cmd])
print('cd ' + dir_name + ' && ' + bash_cmd)
os.system('cd ' + dir_name + ' && ' + bash_cmd)

logging.info("finish all training")

# # copy output data
# s3_output_dir = args.tensor_dir
# logging.info("copy local data to s3")
# logging.info(mox.file.list_directory(local_data_dir_output, recursive=True))
# # s3_output_dir=os.path.join(os.path.join(args.data_url, task),args.output_dir)

# print('output dir:' + s3_output_dir)
# mox.file.copy_parallel(local_data_dir_output, s3_output_dir)
