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
from tqdm import *

nlp = spacy.load("es_dep_news_trf", exclude=['morphologizer', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
print('loading nlp succesfully')

def spacy_tokenizer(document, nlp=nlp):
    # nlp = spacy.load("es_dep_news_trf", exclude=['morphologizer', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
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
    
    total_tokens = []
    count = 0
    
    long_sentences_list = []
    long_seq = ''
    for line in sp_text:
        long_seq += line
        if count % 500 == 0 or count == len(sp_text)-1:
            long_sentences_list.append(long_seq)
            # print('current long_seq: ', long_seq)
            long_seq = ''
        count += 1
            
    print(len(long_sentences_list))
    print(long_sentences_list[2])
    
    for line in tqdm(long_sentences_list):
        
        new_tokens = list(spacy_tokenizer(line))
        print('new_tokens: ', new_tokens[:20])
        total_tokens+=new_tokens

    total_tokens = list(set(total_tokens))
    
    print('The numbers of total spanish tokens are: ', len(total_tokens))
    
    return total_tokens

### main_function #########################################################

spanish_json = osp.join(os.path.abspath('.'), "multiling_corpus/spanish_med.json")
spanish_lists = []
with open(spanish_json, 'r', encoding='utf-8') as sp_data:
    sp_data = json.load(sp_data)
    for v in sp_data:
        line = v['tokens']
        spanish_lists.append(line)

print('len spanish lists ', len(spanish_lists))


model_path = osp.join(os.path.abspath('.'), "models/CXRBert_general")
spanish_sample = osp.join(os.path.abspath('.'), "spanish_sample/spanish_sample.txt")

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

documents = spanish_lists


spanish_tokens = add_new_tokens(documents) 

new_tokens = []
for token in spanish_tokens:
    if token not in tokenizer.vocab.keys():
        new_tokens.append(token)

# new_tokens = set(spanish_tokens) - set(tokenizer.vocab.keys())
# save new tokens to vacab.txt
sp_vocab = '/home/intern2/Brucewan/Multilingual_CXRBert/spacy_spanish/sp_random_vocab.txt'

with open(sp_vocab, 'w', encoding='utf-8') as out_data:
    for token in new_tokens:
        out_data.write(''.join([line]))
        out_data.write('\n')
out_data.close()

print('new_token after filtering with original vocabs ', new_tokens)
tokenizer.add_tokens(list(new_tokens))
emb_before = model.bert.embeddings.word_embeddings

print('word embedding shape before ', emb_before)
model.resize_token_embeddings(len(tokenizer))
emb_before = model.bert.embeddings.word_embeddings
print('word embedding shape after ', emb_before)
    
    
    

    
    