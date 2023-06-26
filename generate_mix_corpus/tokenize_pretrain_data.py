import os
import os.path as osp
import pandas as pd
from transformers import AutoConfig, AutoTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer as NewBertTokenizer
from transformer.modeling_cxrbert import BertForMaskedLM
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json



if __name__ == '__main__':
    
    ######################## loading model test ###################################
    model_path = osp.join(os.path.abspath('.'), "models/CXRBert_general")
    tokenizer = NewBertTokenizer.from_pretrained(model_path, do_lower_case=True)
    config = AutoConfig.from_pretrained(
        model_path + '/config.json',
        cache_dir=None,
        )

    model = BertForMaskedLM.from_pretrained(
        model_path + '/pytorch_model.bin', 
        from_tf=bool(".ckpt" in model_path + '/pytorch_model.bin'),
        config=config, 
        cache_dir=None)

    print('Loading cxrbert successfully')
    vocab_list = list(tokenizer.vocab.keys())
    print("tokenizer vocab len before extending new tokens: ", len(vocab_list))
    
    emb_before = model.bert.embeddings.word_embeddings

    print('word embedding shape before ', emb_before)
    model.resize_token_embeddings(len(tokenizer))
    emb_before = model.bert.embeddings.word_embeddings
    print('word embedding shape after ', emb_before)
    
    ###################### Loading model test ######################################
    
    
    mix_json = osp.join(os.path.abspath('.'), "multiling_corpus/mix_corpus.json")
    mix_tokens_corpus = osp.join(os.path.abspath('.'), "multiling_corpus/mix_tokens_corpus.json")
    
    mix_out = [{}]
    with open(mix_json, 'r', encoding='utf-8') as mix_data, open(mix_tokens_corpus, 'w', encoding='utf-8') as out_data:

        mix_data = json.load(mix_data)
        attributes = ["tokens"]
        for v in mix_data:
            line = v['tokens']
            token_line = tokenizer.tokenize(line)
            token_lein = ['[CLS]'] + token_line + ['[SEP]']
            print('token line', token_line)
            new_values = [token_line]
            entry = dict(zip(attributes, new_values))
            mix_out.append(entry)
        json.dump(mix_out[1:],out_data )
   
    
    