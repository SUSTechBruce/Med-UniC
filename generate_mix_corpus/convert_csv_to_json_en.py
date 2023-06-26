import pandas as pd
import os.path as osp
import os
import json
import codecs


if __name__ == '__main__':
    en_csv = osp.join(os.path.abspath('.'), "multiling_corpus/MIMIC-CXR-meta_INDI_FIND_IMP_report.csv")
    fingdings =  list(pd.read_csv(en_csv)['FINDINGS'])
    impression = list(pd.read_csv(en_csv)['IMPRESSION'])
    indication = list(pd.read_csv(en_csv)['INDICATION'])
    
    spanish_out = osp.join(os.path.abspath('.'), "multiling_corpus/en_med.json")
    
    data = [{}]
    attributes = ["tokens"]
    
    nan_count = 0
    
    line_len = len(fingdings)
    with open(spanish_out, 'w') as output_file:
        
        for i in range(line_len):
            # print("*************", line)
            try:    
                contexts = ''.join(fingdings[i] + impression[i] + indication[i])
                print(contexts)
                new_values = [contexts]
                entry = dict(zip(attributes, new_values))
                data.append(entry)
            except:
                nan_count += 1
                print('The wrong line', line)
                
        print('wrong nan value: ', nan_count)
        json.dump(data[1:], output_file)
        

            
            
            
            
        
        
        
        
        
            
            
        
    
    
    

    
