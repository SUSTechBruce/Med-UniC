import pandas as pd
import os.path as osp
import os
import json
import codecs


if __name__ == '__main__':
    spanish_csv = osp.join(os.path.abspath('.'), "multiling_corpus/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv")
    spanish_corpus =  pd.read_csv(spanish_csv)['Report']
    spanish_corpus = list(spanish_corpus)
    print(spanish_corpus[0])
    
    spanish_out = osp.join(os.path.abspath('.'), "multiling_corpus/spanish_med.json")
    
    data = [{}]
    attributes = ["tokens"]
    
    nan_count = 0
    
    with open(spanish_out, 'w') as output_file:
        
        for line in spanish_corpus:
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
        
        
    # loading json file into txt corpus file
    spanish_txt = osp.join(os.path.abspath('.'), "multiling_corpus/spanish_corpus.txt")
    with open(spanish_out, 'r', encoding='utf-8') as data,  open(spanish_txt, 'w', encoding='utf-8') as out_data:
        json_data = json.load(data)
        for v in json_data:
            line = v['tokens']
            print('line: ', line)
            out_data.write(''.join([line]))
            out_data.write('\n')
        
    out_data.close()
    
    print('save to txt successfully')
            
            
            
            
        
        
        
        
        
            
            
        
    
    
    

    
