# -*- coding:utf-8 -*-

import moxing as mox
import os
import argparse
import logging

os.environ["NCCL_DEBUG"] = "INFO"

parser = argparse.ArgumentParser()
parser.add_argument('--init_method', type=str, default=None, help='cloud will handle this, do not set any value by yourself')
parser.add_argument('--rank', type=int, default=0, help='cloud will handle this, do not set any value by yourself')
parser.add_argument('--world_size', type=int, default=1, help='cloud will handle this, do not set any value by yourself')
parser.add_argument('--data_url', type=str, default="/data/zhangwei/ict_protein/output/protein_seq_input_ids", help='data dir on s3')


parser.add_argument('--nproc_per_node', type=int, default=0, help='cloud will handle this, do not set any value by yourself')
parser.add_argument('--nnodes', type=int, default=1, help='cloud will handle this, do not set any value by yourself')
parser.add_argument('--pregenerated_data', type=str, required=True, default='/nas/hebin/data/english-exp/books_wiki_tokens_ngrams')
parser.add_argument("--gpt_model", type=str, default='/nas/hebin/data/english-exp/models/bert-base-uncased/', help="")
# parser.add_argument('--output_dir', type=str, required=True, default='s3://bucket-373/hebin/code/zen/mwe/output')
# parser.add_argument("--tsbd_dir", type=str, default='/home/ma-user/work/zen/mwe/output/tsbd', help="tensorflow dir")
parser.add_argument('--nas_output_dir', type=str, required=True, default='s3://bucket-375/hebin/code/zen/mwe/output', help='')
parser.add_argument('--cache_dir', type=str, default='/cache', help='')

parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train for")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--train_batch_size",
                    default=16,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--learning_rate",
                    default=1e-4,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument('--apex_folder', type=str, default='s3://bucket-373/hebin/tools/apex/')
parser.add_argument('--max_ngram_in_sequence', type=int, default=48)
parser.add_argument('--fusion_layer', type=str, default='first')
parser.add_argument("--ngram_ratio",default=1.0,type=float,help="")
parser.add_argument("--mask_ratio", default=0.15, type=float, help="")
parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--model', type=str, required=True)
parser.add_argument("--model_type", default='bert', type=str)
parser.add_argument("--load_model", default=0, type=int)
parser.add_argument("--warmup_steps", default=10000, type=int)
parser.add_argument('--tensor_dir', type=str, default='s3://bucket-373/hebin/tools/apex/')
# prefix_token_len
args, _ = parser.parse_known_args()


################################# copy generate_data to mox ##############################
DS_DIR_NAME = "src"
os.environ['DLS_LOCAL_CACHE_PATH'] = "/cache"

LOCAL_DIR = os.environ['DLS_LOCAL_CACHE_PATH']
assert mox.file.exists(LOCAL_DIR)
logging.info("local disk: " + LOCAL_DIR)

dir_name = os.path.dirname(os.path.abspath(__file__))

# copy data to local /cache/lcqmc
logging.info("copying data...")
local_dir = os.path.join(LOCAL_DIR, DS_DIR_NAME)

# copy data to local /cache/lcqmc
logging.info("copying data...")
local_data_dir = os.path.join(LOCAL_DIR, DS_DIR_NAME)
logging.info(mox.file.list_directory(args.pregenerated_data, recursive=True))
mox.file.copy_parallel(args.pregenerated_data, local_data_dir)

# copy valid data to local /cache/lcqmc


print('############################# copy pregenerated data ##############################')
#############################################

host, port = args.init_method[:-5], args.init_method[-4:]
print(host, port)

init_method = args.init_method
rank = str(args.rank)
world_size = str(args.world_size)
print(init_method, rank, world_size)

# try:
#     import apex
# except Exception:
print('Copy apex start...')
# copy apex-master folder to local machine
mox.file.copy_parallel(args.apex_folder, '/cache/apex-master')
os.system('python -m pip install --upgrade --force pip ')
os.system('python -m pip install transformers==4.9.0 ')
os.system('python -m pip install scikit-learn ')
os.system('python -m pip install spacy ')
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

script = os.path.join(dir_name, 'run_pretraining_cxrbert.sh')

cmd = ['bash', script, host, port, init_method, rank, args.nnodes, local_data_dir,
       args.nas_output_dir, args.cache_dir, args.epochs, args.gradient_accumulation_steps,
       args.train_batch_size, args.learning_rate, args.max_seq_length,
       args.model, args.model_type, args.load_model, args.nproc_per_node,
       args.warmup_steps, args.mask_ratio, args.tensor_dir]

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
