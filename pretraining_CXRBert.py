# coding: utf-8
# Copyright 2019 Sinovation Ventures AI Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch pretrain for ZEN model."""

import os
import os.path as osp


from argparse import ArgumentParser
from pathlib import Path


import json
import random
import numpy as np
from collections import namedtuple
import time
import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle
import collections

# This is used for running on Huawei Cloud.
oncloud = True
try:
    import moxing as mox
except:
    oncloud = False

from transformer.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME

from transformer.modeling_cxrbert import BertForMaskedLM

from transformers import AutoConfig, AutoTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer as NewBertTokenizer

from gpt.modeling import GPT2LMHeadModel
from transformer.tokenization import BertTokenizer
from gpt.tokenization import GPT2Tokenizer
from transformer.optimization import AdamW, get_linear_schedule_with_warmup, BertAdam

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import logging

from apex.parallel import DistributedDataParallel as DDP

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

InputFeatures = namedtuple(
    "InputFeatures",
    "input_ids input_mask lm_label_ids ")

GPTInputFeatures = namedtuple(
    "InputFeatures",
    "input_ids")

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))

    try:
        mask_indices = sorted(random.sample(range(1, len(tokens)-1), num_to_mask))
    except:
        print('Found error in tokens: {}'.format(tokens))
        return None, None, None

    masked_token_labels = [tokens[index] for index in mask_indices]
    for index in mask_indices:
        masked_token = None
        # 80% of the time, replace with [MASK]
        if random.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = random.choice(vocab_list)
        tokens[index] = masked_token

    return tokens, mask_indices, masked_token_labels



def convert_example_to_features(example, args):
    tokens1 = example["tokens"] # context

    if args.model_type == 'gpt':
        if len(tokens1) > args.max_seq_length:
            print('Assertion Warning!')
            print(f'{len(tokens1)}')  # The preprocessed data should be already truncated
        
        try:
            input_ids = args.tokenizer.convert_tokens_to_ids(tokens1)
            input_array = -np.ones(args.max_seq_length, dtype=np.int)
            input_array[:len(input_ids)] = input_ids
        except Exception as e:
            print(e)
            print(tokens1)

        features = GPTInputFeatures(input_ids=input_array)
        return features
    
    if len(tokens1) >= args.max_seq_length:
        tokens1 = tokens1[:args.max_seq_length]

    tokens1[0], tokens1[-1] = '[CLS]', '[SEP]'
    
    # start to split the case_label
    
    tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions( tokens1, args.masked_lm_prob,args.max_predictions_per_seq, args.vocab_list)

    if tokens is None:
        return None

    assert len(tokens) <= args.max_seq_length  # The preprocessed data should be already truncated
    try:
        input_ids = args.tokenizer.convert_tokens_to_ids(tokens)
        masked_label_ids = args.tokenizer.convert_tokens_to_ids(masked_lm_labels)
    except Exception as e:
        print(e)
        print(tokens1)
        print(tokens)
        print(masked_lm_labels)

    input_array = np.zeros(args.max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(args.max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    lm_label_array = np.full(args.max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids
    

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             # segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                            
                            )
    return features


def mask_and_choose(batch, num_samples, args):
    seq_len = args.max_seq_length
    input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)

    # gpt 不要做MASK
    if args.model_type == 'gpt':
        for i, line in enumerate(batch):
            example = json.loads(line)
            features = convert_example_to_features(example, args)
            input_ids[i] = features.input_ids

        input_ids = torch.from_numpy(input_ids.astype(np.int64))
        return input_ids
        

    input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
    lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)

    for i, line in enumerate(batch):
        example = line
        
        # print(example)
        features = convert_example_to_features(example, args)

        if features is None:
            continue

        input_ids[i] = features.input_ids
        input_masks[i] = features.input_mask
        lm_label_ids[i] = features.lm_label_ids


    input_ids = torch.from_numpy(input_ids.astype(np.int64))
    input_masks = torch.from_numpy(input_masks.astype(np.int64))
    lm_label_ids = torch.from_numpy(lm_label_ids.astype(np.int64))
    
    return (input_ids, input_masks, lm_label_ids)


def run_task(data_files, args):
    start_time = time.time()
    logging.info('Running thread %s, %s files', args.rank, len(data_files))
    for i, data_file in enumerate(data_files):
        input_data_file = os.path.join(args.pregenerated_data, data_file)
        logging.info("Loading inputs from file %s", input_data_file)
        examples = []
                
        with open(input_data_file, 'r', encoding='utf-8') as mix_data:
            mix_data = json.load(mix_data)
            for example_line in tqdm(mix_data, desc="Training examples"):

                examples.append(example_line)

        
        print('len example #################', len(examples))
        if args.debug:
            examples = examples[:20000]  # debug

        logging.info("num_samples before cut %s", len(examples))
        num_samples = int(len(examples)/args.world_size)
        if args.model_type == 'bert':
            input_ids, input_masks, lm_label_ids = mask_and_choose(examples[args.rank*num_samples:(args.rank+1)*num_samples], num_samples, args)
        elif args.model_type == 'gpt':
            input_ids = mask_and_choose(examples[args.rank*num_samples:(args.rank+1)*num_samples], num_samples, args)

        logging.info("num_samples after cut %s", num_samples)

        # for k in range(int(hvd.size())):
        data_file_cached = os.path.join(args.local_data_dir, data_file + '.cached.' + str(args.rank))
        logging.info("cached file %s", data_file_cached)
        with open(data_file_cached, "wb") as handle:
            if args.model_type == 'bert':
                pickle.dump([input_ids, input_masks, lm_label_ids], handle, protocol=pickle.HIGHEST_PROTOCOL)
            elif args.model_type == 'gpt':
                pickle.dump(input_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info('%s/%s processed in thread %s, time cost is %.2f secs' % (i + 1, len(data_files), args.rank, time.time() - start_time))


def load_doc_tokens_ngrams(args):
    data_files = []
    for inputfile in os.listdir(args.pregenerated_data):
        input_file = os.path.join(args.pregenerated_data, inputfile)
        if os.path.isfile(input_file) and inputfile.startswith('mix_tokens'):
            data_files.append(inputfile)

    file_count = len(data_files)
    print('The length of file_count ################ : ', file_count)
    run_task(data_files, args)

    t_input_ids, t_input_masks, t_lm_label_ids = [], [], []
    for i in range(file_count):
        data_file_cached = os.path.join(args.local_data_dir, data_files[i] + '.cached.' + str(args.rank))
        with open(data_file_cached, "rb") as handle:
            input_ids, input_masks, lm_label_ids = pickle.load(handle)

        logging.info("Loading inputs from cached file %s", data_file_cached)
        logging.info("num_samples %s", len(input_ids))

        if i == 0:
            t_input_ids, t_input_masks, t_lm_label_ids = [input_ids], [input_masks], [lm_label_ids]
        else:
            t_input_ids.append(input_ids)
            t_input_masks.append(input_masks)
            t_lm_label_ids.append(lm_label_ids)
            
        logger.info("Dataset %s loaded", data_file_cached)
    t_input_ids = torch.cat(t_input_ids, 0)
    t_input_masks = torch.cat(t_input_masks, 0)
    t_lm_label_ids = torch.cat(t_lm_label_ids, 0)

    
    print('t_lm_label_ids shape', t_lm_label_ids.shape)
    
    logging.info("total num_samples %s", len(t_input_ids))
    for i in range(3):
        logging.info("*** Example ***")
        logging.info("block %s" % i)
        tokens = args.tokenizer.convert_ids_to_tokens(t_input_ids[i].tolist())
        logging.info("inputs: %s" % ' '.join([str(item) for item in tokens]))
        logging.info("input_masks: %s" % ' '.join([str(item) for item in t_input_masks[i].tolist()]))
        logging.info("lm_label_ids: %s" % ' '.join([str(item) for item in t_lm_label_ids[i].tolist()]))


    dataset = TensorDataset(t_input_ids, t_input_masks, t_lm_label_ids)
    return dataset


def load_doc_tokens_ngrams_gpt(args):
    data_files = []
    for inputfile in os.listdir(args.pregenerated_data):
        input_file = os.path.join(args.pregenerated_data, inputfile)
        if os.path.isfile(input_file) and inputfile.endswith('json') and inputfile.startswith('train_doc_tokens_ngrams'):
            data_files.append(inputfile)

    file_count = len(data_files)
    run_task(data_files, args)

    t_input_ids, t_input_masks, t_lm_label_ids = [], [], []
    for i in range(file_count):
        data_file_cached = os.path.join(args.local_data_dir, data_files[i] + '.cached.' + str(args.rank))
        with open(data_file_cached, "rb") as handle:
            input_ids = pickle.load(handle)

        logging.info("Loading inputs from cached file %s", data_file_cached)
        logging.info("num_samples %s", len(input_ids))

        if i == 0:
            t_input_ids = [input_ids]
        else:
            t_input_ids.append(input_ids)
        logger.info("Dataset %s loaded", data_file_cached)
    t_input_ids = torch.cat(t_input_ids, 0)
    logging.info("total num_samples %s", len(t_input_ids))
    for i in range(1):
        logging.info("*** Example ***")
        logging.info("block %s" % i)
        tokens = args.tokenizer.convert_ids_to_tokens(t_input_ids[i].tolist())
        logging.info("inputs: %s" % ' '.join([str(item) for item in tokens]))

    dataset = TensorDataset(t_input_ids)
    return dataset



def spacy_tokenizer(document):
    nlp = spacy.load("es_dep_news_trf", exclude=['morphologizer', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
    # tokenize the document with spaCY
    doc = nlp(document)
    # Remove stop words and punctuation symbols
    tokens = [
        token.text for token in doc if (
        token.is_stop == False and \
        token.is_punct == False and \
        token.text.strip() != '' and \
        token.text.find("\n") == -1)]
    return tokens

def add_new_tokens(sp_text):
    # intialize the tokenizer with Spanish Spacy 

    # apply spacy tokenizer with sklearn
    tfidf_vectorizer = TfidfVectorizer(lowercase=False, tokenizer=spacy_tokenizer, 
                                       norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    
    # parse matrix of tfidf
    length = len(sp_text)
    result = tfidf_vectorizer.fit_transform(sp_text)  
    
    print('Tf-idf vector shape: ', result.shape)
    
    # get idf of tokens
    idf = tfidf_vectorizer.idf_
    
    # get tokens from most frequent in documents to least frequent
    idf_sorted_indexes = sorted(range(len(idf)), key=lambda k: idf[k])
    idf_sorted = idf[idf_sorted_indexes]
    tokens_by_df = np.array(tfidf_vectorizer.get_feature_names())[idf_sorted_indexes]
    new_tokens = tokens_by_df
    
    return new_tokens
    
    

def main():
    parser = ArgumentParser()
    parser.add_argument('--pregenerated_data', type=str, required=True, default='/nas/hebin/data/english-exp/books_wiki_tokens_ngrams')
    parser.add_argument('--nas_output_dir', type=str, required=True, default='s3://bucket-375/hebin/code/zen/mwe/output')
    parser.add_argument('--cache_dir', type=str, default=None, help='')
    parser.add_argument('--model', type=str, default='8layer_student', required=True)
    parser.add_argument('--data_url', type=str, default="/data/zhangwei/ict_protein/output/protein_seq_input_ids", help='data dir on s3')

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

    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--debug',
                        action='store_true',
                        help="Whether to debug")

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--masked_lm_prob", type=float, default=0.0,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=77,
                        help="Maximum number of tokens to mask in each sequence")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--logging_steps", type=int, default=5, help="Log every X updates steps.")
    parser.add_argument("--warmup_steps", default=10000, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--model_type", default='bert', type=str)
    parser.add_argument("--load_model", default=0, type=int)
    parser.add_argument('--tensor_dir', type=str, default=osp.join(os.path.abspath('.'), "labels/all_reason_names.txt"), help='data dir on s3')

    

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
    device = torch.device("cuda", args.local_rank)
    print('device_id: %s' % args.local_rank)
    print('device_count: %s, rank: %s, world_size: %s' % (device_count, args.rank, args.world_size))
    print(init_method)

    torch.distributed.init_process_group(backend='nccl', world_size=args.world_size,
                                         rank=args.rank, init_method=init_method)

    LOCAL_DIR = args.nas_output_dir
    if args.rank == 0:
        if not os.path.exists(LOCAL_DIR):
            os.makedirs(LOCAL_DIR)
            logger.info(LOCAL_DIR + ' created!')
    
    # assert mox.file.exists(LOCAL_DIR)

    if args.local_rank == 0:
        print('Moxing successfully #################')
        logging.info(mox.file.list_directory(args.pregenerated_data, recursive=True))
        logging.info(mox.file.list_directory(args.model, recursive=True))

    local_save_dir = os.path.join(LOCAL_DIR, 'output', 'bert', 'checkpoints')
    tensor_dir = os.path.join(LOCAL_DIR, 'output', 'bert', 'tensorboard')
    
    # tensor_dir = os.path.join(args.tensor_dir, 'tensorboard')

    save_name = '_'.join([
        '{}'.format(args.model_type),
        'epoch', str(args.epochs),
        'lr', str(args.learning_rate),
        'bsz', str(args.train_batch_size),
        'grad_accu', str(args.gradient_accumulation_steps),
        str(args.max_seq_length),
        'gpu', str(args.world_size),
    ])
    bash_save_dir = os.path.join(local_save_dir, save_name)
    bash_tsbd_dir = os.path.join(tensor_dir, save_name)
    if args.rank == 0:
        if not os.path.exists(bash_save_dir):
            os.makedirs(bash_save_dir)
            logger.info(bash_save_dir + ' created!')
        if not os.path.exists(bash_tsbd_dir):
            os.makedirs(bash_tsbd_dir)
            logger.info(bash_tsbd_dir + ' created!')

    local_data_dir_tmp = '/cache/data/tmp/'
    local_data_dir = local_data_dir_tmp + save_name

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
        

    if args.model_type == 'bert':
        
        ############# Loading the transformers.BertTokenizer ####################
        args.tokenizer = NewBertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
        vocab_list = list(args.tokenizer.vocab.keys())
        
        print("tokenizer vocab len before extending new tokens: ", len(vocab_list))
        
        ############# Loading the AutoConfig ###################################
        config = AutoConfig.from_pretrained(
        args.model + '/config.json',
        cache_dir=None,
        )
        
        ############# Loading transformers BertModel ##########################
        if args.load_model:
            model = BertForMaskedLM.from_pretrained(
                args.model + '/pytorch_model.bin', 
                from_tf=bool(".ckpt" in args.model + '/pytorch_model.bin'),
                config=config, 
                cache_dir=None)
            
        else:
            model = BertForMaskedLM.from_scratch(args.model)
    elif args.model_type == 'gpt':
        args.tokenizer = GPT2Tokenizer.from_pretrained(args.model)
        args.vocab_list = list(args.tokenizer.encoder.keys())
        model = GPT2LMHeadModel.from_scratch(args.model)
    

    model.resize_token_embeddings(len(args.tokenizer))
    
    emb_before = model.bert.embeddings.word_embeddings
    print('word embedding shape after ####################### ', emb_before)
    
    
    args.vocab_list = list(args.tokenizer.vocab.keys())
    
    print("tokenizer vocab len after extending new tokens################: ", len(args.vocab_list))
    
    model.to(device)

    if args.local_rank == 0:
        tb_writer = SummaryWriter(bash_tsbd_dir)

    global_step = 0
    step = 0
    tr_loss, logging_loss = 0.0, 0.0
    
    end_time, start_time = 0, 0

    for epoch in range(args.epochs):
        args.local_data_dir = os.path.join(local_data_dir, str(epoch))
        if args.local_rank == 0:
            os.makedirs(args.local_data_dir)
        
        while 1:
            if os.path.exists(args.local_data_dir):
                if args.model_type == 'bert':
                    epoch_dataset = load_doc_tokens_ngrams(args)
                elif args.model_type == 'gpt':
                    epoch_dataset = load_doc_tokens_ngrams_gpt(args)
                break
            print('Dead loop please check ##############')

        if args.local_rank == 0:
            logging.info('Dataset in epoch %s', epoch)
            logging.info(mox.file.list_directory(args.local_data_dir, recursive=True))

        # rank = 0
        train_sampler = DistributedSampler(epoch_dataset, num_replicas=1, rank=0)
        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        step_in_each_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_train_optimization_steps = step_in_each_epoch * args.epochs
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(epoch_dataset) * args.world_size)
        logger.info("  Num Epochs = %d", args.epochs)
        logging.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                     args.train_batch_size * args.gradient_accumulation_steps * args.world_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logging.info("  Num steps = %d", num_train_optimization_steps)

        # Prepare optimizer
        if epoch == 0:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            warm_up_ratio = args.warmup_steps / num_train_optimization_steps
            print('warm_up_ratio: {}'.format(warm_up_ratio))
            optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate,
                                    e=args.adam_epsilon, schedule='warmup_linear',
                                    t_total=num_train_optimization_steps,
                                    warmup=warm_up_ratio)

            if args.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError("Please install apex from https://www.github.com/nvidia/apex"
                                        " to use fp16 training.")
                model, optimizer = amp.initialize(model, optimizer,
                                                            opt_level=args.fp16_opt_level,
                                                            min_loss_scale=1) #
            # apex
            model = DDP(model, message_size=10000000,
                                gradient_predivide_factor=torch.distributed.get_world_size(),
                                delay_allreduce=True)
            logger.info('apex data paralleled!')

        model.train()
        for step_, batch in enumerate(train_dataloader):
            step += 1
            batch = tuple(t.to(device) for t in batch)

            if args.model_type == 'bert':
                input_ids, input_masks, lm_label_ids = batch
                # using CXRBert to pretrain
                loss, _, _, _ = model(input_ids, attention_mask=input_masks, labels=lm_label_ids)
            elif args.model_type == 'gpt':
                input_ids = batch
                loss = model(input_ids)

            tr_loss += loss.item()

            
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
            else:
                loss.backward(retain_graph=True)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if (step + 1) % (args.gradient_accumulation_steps * args.logging_steps) == 0 \
                        and args.local_rank < 2 or global_step < 100:
                    end_time = time.time()

                    logger.info(
                            'Epoch: %s, global_step: %s/%s, lr: %s, loss is %s; '
                            ' (%.2f sec)' %
                            (epoch, global_step + 1, step_in_each_epoch, optimizer.get_lr()[0],
                                loss.item() * args.gradient_accumulation_steps,
                                end_time - start_time))
                    
                    start_time = time.time()

                if args.logging_steps > 0 and global_step % args.logging_steps == 0 and args.local_rank == 0:
                    tb_writer.add_scalar("lr", optimizer.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps / args.gradient_accumulation_steps, global_step)
                    
                    logging_loss = tr_loss



        # Save a trained model
        if args.rank == 0:
            saving_path = bash_save_dir

            saving_path = Path(os.path.join(saving_path, "epoch_" + str(epoch)))

            if saving_path.is_dir() and list(saving_path.iterdir()):
                logging.warning(f"Output directory ({ saving_path }) already exists and is not empty!")
            saving_path.mkdir(parents=True, exist_ok=True)

            logging.info("** ** * Saving fine-tuned model ** ** * ")
            model_to_save = model.module if hasattr(model, 'module')\
                else model  # Only save the model it-self

            output_model_file = os.path.join(saving_path, WEIGHTS_NAME)
            output_config_file = os.path.join(saving_path, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            args.tokenizer.save_vocabulary(saving_path)

            torch.save(optimizer.state_dict(), os.path.join(saving_path, "optimizer.pt"))
            logger.info("Saving optimizer and scheduler states to %s", saving_path)

    if args.local_rank == 0:
        tb_writer.close()


if __name__ == '__main__':
    main()
