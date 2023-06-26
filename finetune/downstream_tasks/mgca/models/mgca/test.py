from model_MedKLIP import MedKLIP
import ruamel_yaml as yaml
import os
import os.path as osp
import torch


def get_tokenizer(tokenizer,target_text):
    
    target_tokenizer = tokenizer(list(target_text), padding='max_length', truncation=True, max_length=128,return_tensors="pt")
    
    return target_tokenizer

if __name__ == '__main__':
    config = osp.join(os.path.abspath('.'), "Pretrain_MedKLIP.yaml")
    config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
                                                        
    print("Creating model")
    model = MedKLIP(config,None, None, mode = 'train')
    device = 'cuda'
    model = model.to(device)
    print('Start to load model')
    
    
    checkpoint_path =  osp.join(os.path.abspath('.'), "baseline_models/MedKLIP/checkpoint_final.pth")                             
    checkpoint = torch.load(checkpoint_path, map_location='cpu') 
    state_dict = checkpoint['model']
    
    
    weights_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
        

    model.load_state_dict(weights_dict)
    
    print('loading ok')
    
    


        
                                                                      
                                                                      
                                                                      
    
                                                                      
