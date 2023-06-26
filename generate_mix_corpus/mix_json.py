import pandas as pd
import os.path as osp
import os
import json
import codecs
import random

if __name__ == '__main__':
    
    spanish_data = osp.join(os.path.abspath('.'), "multiling_corpus/spanish_med.json")
    en_data = osp.join(os.path.abspath('.'), "multiling_corpus/en_med.json")
    
    mix_out = osp.join(os.path.abspath('.'), "multiling_corpus/mix_corpus.json")
    
    spanish_lists = []
    en_lists = []
    
    with open(spanish_data, 'r', encoding='utf-8') as sp_data,  open(en_data, 'r', encoding='utf-8') as en_data:
        sp_data = json.load(sp_data)
        en_data = json.load(en_data)
        for v in sp_data:
            line = v['tokens']
            spanish_lists.append(line)
            
        for v in en_data:
            line = v['tokens']
            en_lists.append(line)
            
    total_list = en_lists + spanish_lists
    random.shuffle(total_list)
    ###### write to new tokens ######
    
    data = [{}]
    attributes = ["tokens"]
    
    nan_count = 0
    
    with open(mix_out, 'w') as output_file:
        
        for line in total_list:
            # print("*************", line)
            try:
                contexts = ''.join(line)
                new_values = [contexts]
                entry = dict(zip(attributes, new_values))
                data.append(entry)
            except:
                nan_count += 1
                print('The wrong line', line)
                
        print('wrong nan value: ', nan_count)
        json.dump(data[1:], output_file)
    
    print('save to mix json successfully')
            
            
            
            
        
        
        
        
        
            
            
        
    
    
    

    
