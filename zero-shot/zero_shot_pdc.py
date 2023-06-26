

import os
from pathlib import Path
from typing import List, Tuple, Optional#

import subprocess
import numpy as np
import sys
# sys.path.append('/home/cl522/reproduce_work/cross_lingual_multimodal_codes/utils')
import utils_builder
import os.path as osp
import os 
import pandas as pd
from PIL import Image
import h5py
import matplotlib.pyplot as plt

import torch
from torch.utils import data
from tqdm.notebook import tqdm
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from torchvision.transforms import transforms
import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import average_precision_score

from argparse import ArgumentParser
import random
import numpy as np
import pandas as pd
from PIL import Image
def sigmoid(x): 
    z = 1/(1 + np.exp(-x)) 
    return z

class CXRTestDataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        img_path: Path to hdf5 file containing images.
        label_path: Path to file containing labels 
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(
        self, 
        img_path: str, 
        transform = None, 
    ):
        super().__init__()
        self.img_dset = h5py.File(img_path, 'r')['cxr']
        self.transform = transform
            
    def __len__(self):
        return len(self.img_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.img_dset[idx] # np array, (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img) # torch, (320, 320)
        
        if self.transform:
            img = self.transform(img)
            
        sample = {'img': img}
    
        return sample
    
    
class PDC_Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        img_path: Path to hdf5 file containing images.
        label_path: Path to file containing labels 
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(
        self, 
        img_path: str, 
        transform = None, 
    ):
        super().__init__()
        print('start to loading PDC_Dataset for zeroshot')
        self.img_dset = np.load(img_path, allow_pickle=True, mmap_mode='r')
        pdc_zeroshot = osp.join(os.path.abspath('.'), "test_sample_sp.csv")
        pdc_zeroshot = pd.read_csv(pdc_zeroshot)['img_id'].tolist()
        print('pdc_zeroshot', pdc_zeroshot[:25])
        
        print('ok to loading PDC_Dataset for zeroshot')
        self.img_dset = self.img_dset[pdc_zeroshot]

        
        self.transform = transform
            
    def __len__(self):
        return len(self.img_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = self.img_dset[idx]
        img = Image.fromarray(img).convert("RGB")
            
# #         img = self.img_dset[idx] # np array, (320, 320)
#         img = np.expand_dims(img, axis=0)
#         img = np.repeat(img, 3, axis=0)
# #         img = torch.from_numpy(img) # torch, (320, 320)

        
        if self.transform:
            img = self.transform(img)
            
        sample = {'img': img}
    
        return sample
    
    
def zeroshot_classifier(classnames, templates, model, context_length=77):
    """
    FUNCTION: zeroshot_classifier
    -------------------------------------
    This function outputs the weights for each of the classes based on the 
    output of the trained clip model text transformer. 
    
    args: 
    * classnames - Python list of classes for a specific zero-shot task. (i.e. ['Atelectasis',...]).
    * templates - Python list of phrases that will be indpendently tested as input to the clip model.
    * model - Pytorch model, full trained clip model.
    * context_length (optional) - int, max number of tokens of text inputted into the model.
    
    Returns PyTorch Tensor, output of the text encoder given templates. 
    """
    with torch.no_grad(): # to(device=torch.device("cuda"iftorch.cuda.is_available()else"cpu")) 
        zeroshot_weights = []
        # compute embedding through model for each class
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] # format with class
            texts = model._tokenize(texts) # tokenize
            class_embeddings = model.lm_model(texts.input_ids.to(device=torch.device("cuda" if torch.cuda.is_available() else"cpu")) 
                                              , texts.attention_mask.to(device=torch.device("cuda"if torch.cuda.is_available()else"cpu")) 
                                             ).last_hidden_state # embed with text encoder
            class_embeddings = model.proj_t(class_embeddings[:, 0].contiguous()) # embed with text encoder

            # normalize class_embeddings
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # average over templates 
            class_embedding = class_embeddings.mean(dim=0) 
            # norm over new averaged templates
            class_embedding /= class_embedding.norm() 
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights

def predict(loader, model, zeroshot_weights, softmax_eval=True, verbose=0): 
    """
    FUNCTION: predict
    ---------------------------------
    This function runs the cxr images through the model 
    and computes the cosine similarities between the images
    and the text embeddings. 
    
    args: 
        * loader -  PyTorch data loader, loads in cxr images
        * model - PyTorch model, trained clip model 
        * zeroshot_weights - PyTorch Tensor, outputs of text encoder for labels
        * softmax_eval (optional) - Use +/- softmax method for evaluation 
        * verbose (optional) - bool, If True, will print out intermediate tensor values for debugging.
        
    Returns numpy array, predictions on all test data samples. 
    """
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            images = data['img'].to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) 

            # predict
            image_features = model.encoder(images) # (1, 2048)
#             print('image_features: ', image_features[:, :10])
            image_features = model.proj_v(image_features) # (1, 512)
            image_features /= image_features.norm(dim=-1, keepdim=True) # (1, 512)

            # obtain logits
            logits = image_features @ zeroshot_weights # (1, num_classes)
            logits = torch.squeeze(logits, 0) # (num_classes,)
        
            if softmax_eval is False: 
                norm_logits = (logits - logits.mean()) / (logits.std())
                logits = sigmoid(norm_logits) 
            
            y_pred.append(logits.cpu().data.numpy())
            
            if verbose: 
                plt.imshow(images[0][0])
                plt.show()
                print('images: ', images)
                print('images size: ', images.size())
                
                print('image_features size: ', image_features.size())
                print('logits: ', logits)
                print('logits size: ', logits.size())
         
    y_pred = np.array(y_pred)
    return np.array(y_pred)

def run_single_prediction(cxr_labels, template, model, loader, softmax_eval=True, context_length=77): 
    """
    FUNCTION: run_single_prediction
    --------------------------------------
    This function will make probability predictions for a single template
    (i.e. "has {}"). 
    
    args: 
        * cxr_labels - list, labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * template - string, template to input into model. 
        * model - PyTorch model, trained clip model
        * loader - PyTorch data loader, loads in cxr images
        * softmax_eval (optional) - Use +/- softmax method for evaluation 
        * context_length (optional) - int, max number of tokens of text inputted into the model.
        
    Returns list, predictions from the given template. 
    """
    cxr_phrase = [template]
    zeroshot_weights = zeroshot_classifier(cxr_labels, cxr_phrase, model, context_length=context_length)
    y_pred = predict(loader, model, zeroshot_weights, softmax_eval=softmax_eval)
    return y_pred

def run_softmax_eval(model, loader, eval_labels: list, pair_template: tuple, context_length: int = 77): 
    """
    Run softmax evaluation to obtain a single prediction from the model.
    """
     # get pos and neg phrases
    pos = pair_template[0]
    neg = pair_template[1]

    # get pos and neg predictions, (num_samples, num_classes)
    pos_pred = run_single_prediction(eval_labels, pos, model, loader, 
                                     softmax_eval=True, context_length=context_length) 
    neg_pred = run_single_prediction(eval_labels, neg, model, loader, 
                                     softmax_eval=True, context_length=context_length) 

    # compute probabilities with softmax
    sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
    y_pred = np.exp(pos_pred) / sum_pred
    return y_pred, pos_pred, neg_pred
    
def make_true_labels(
    cxr_true_labels_path: str, 
    cxr_labels: List[str],
    cutlabels: bool = True
): 
    """
    Loads in data containing the true binary labels
    for each pathology in `cxr_labels` for all samples. This
    is used for evaluation of model performance.

    args: 
        * cxr_true_labels_path - str, path to csv containing ground truth labels
        * cxr_labels - List[str], subset of label columns to select from ground truth df 
        * cutlabels - bool, if True, will keep columns of ground truth labels that correspond
            with the labels inputted through `cxr_labels`. Otherwise, drop the first column and keep remaining.

    Returns a numpy array of shape (# samples, # labels/pathologies)
        representing the binary ground truth labels for each pathology on each sample.
    """
    # create ground truth labels
    full_labels = pd.read_csv(cxr_true_labels_path)
     
    if cutlabels: 
        full_labels = full_labels.loc[:, cxr_labels]
    else: 
        full_labels.drop(full_labels.columns[0], axis=1, inplace=True)

    y_true = full_labels.to_numpy()
    return y_true

def make(
    model_path: str, 
    cxr_filepath: str, 
    pretrained: bool = True, 
    args = None
):
    """
    FUNCTION: make
    -------------------------------------------
    This function makes the model, the data loader, and the ground truth labels. 
    
    args: 
        * model_path - String for directory to the weights of the trained clip model. 
        * context_length - int, max number of tokens of text inputted into the model. 
        * cxr_filepath - String for path to the chest x-ray images. 
        * cxr_labels - Python list of labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * pretrained - bool, whether or not model uses pretrained clip weights
        * cutlabels - bool, if True, will keep columns of ground truth labels that correspond
        with the labels inputted through `cxr_labels`. Otherwise, drop the first column and keep remaining.
    
    Returns model, data loader. 
    """
    # load model
    ############################
    # just a demo model!
    model = utils_builder.ResNet_CXRBert(args)
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt)
    
    model = model.cuda()
    
    print('loading total checkpoint ok !')
    model.eval()
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
    # load data
    transformations = [
        # means computed from sample in `cxr_stats` notebook
        transforms.ToTensor(),
        # Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        normalize,
    ]
    # if using CLIP pretrained model
    if pretrained: 
        # resize to input resolution of pretrained clip model
        input_resolution = 224
        transformations.append(Resize(input_resolution, interpolation=InterpolationMode.BICUBIC))
    transform = Compose(transformations)
    
    # create dataset
    torch_dset = PDC_Dataset(
        img_path=cxr_filepath,
        transform=transform, 
    )
    loader = torch.utils.data.DataLoader(torch_dset, shuffle=False)
    
    return model, loader



if __name__ == '__main__':
    # official test set 500 samples
    
    torch.manual_seed(2020)
    torch.cuda.manual_seed_all(2020)
    random.seed(2020)
    np.random.seed(2020)
    
    parser = ArgumentParser()
    
    parser.add_argument('--lm_model', type=str, required=False, default='/cxr_bert')
    
    parser.add_argument('--vision_model', type=str, required=False, default='/cxr_bert')
    
    args = parser.parse_args() # pretrain_models/pretrain_mmodal_TF_TT_SLIP_free_9/epoch80/cross-lingual_multi-modal_encoder.pth
    
    args.vision_model = osp.join(os.path.abspath('.'), "../pretrain_models/pretrain_mmodal_TF_TT_SLIP_free_9/epoch80/cross-lingual_multi-modal_encoder.pth")
    args.lm_model = osp.join(os.path.abspath('.'), "lm_model/")
    

    
    cxr_filepath = osp.join(os.path.abspath('.'), "PDC_train_int.npy")
    cxr_true_labels_path = osp.join(os.path.abspath('.'), "test_sample_sp.csv")

    model_path = osp.join(os.path.abspath('.'), "Camp_models/total_unified_loss/epoch80/cross-lingual_multi-modal_total.pth")
    # Multilingual_CXRBert/mmodal_finetune/MGCA/mgca/models/mgca/zero-shot/Camp_models/total_unified_loss_aug/epoch80/cross-lingual_multi-modal_total.pth
    # Multilingual_CXRBert/mmodal_finetune/MGCA/mgca/models/mgca/zero-shot/Camp_models/tf_slip_loss/epoch50/cross-lingual_multi-modal_total.pth
    # Multilingual_CXRBert/mmodal_finetune/MGCA/mgca/models/mgca/zero-shot/Camp_models/clip_loss_3/epoch100/cross-lingual_multi-modal_total.pth
#     cxr_labels: List[str] = ['Atelectasis','Cardiomegaly', 
#                                           'Consolidation', 'Edema','Pleural Effusion', 'Hilar Enlargement', 'Pneumothorax', 'Pneumonia', 'Vascular Redistribution', 'Heart Insufficiency']

    # official 5 labels (chexzero paper only 5 labels)
    
        
    cxr_labels: List[str] = ['atelectasi','cardiomegalia', 
                                          'consolidacion', 'Edem','derramepleural', 'hili prominent', 'neumotorax', 'neumon', 'redistribucion vascul', 'insuficient cardiac']
#     cxr_labels: List[str] = ['Atelectasis','Pleural Effusion', 'Cardiomegaly',  'Pneumonia',  'Heart Insufficiency']
#     cxr_labels: List[str] = ['Atelectasis','Cardiomegaly', 
#                                           'Consolidation', 'Edema','Pleural Effusion', 'Hilar Enlargement', 'Pneumothorax', 'Pneumonia', 'Vascular Redistribution', 'Heart Insufficiency']

    # ---- TEMPLATES ----- # 
    # Define set of templates | see Figure 1 for more details  
#     cxr_pair_template: Tuple[str] = ("{}", "No hay {}"")
#     cxr_pair_template: Tuple[str] = ("{}", "sin hallazg {} relev")
    cxr_pair_template: Tuple[str] = ("{}", "No {}")

    model, loader = make(
            model_path=model_path, 
            cxr_filepath=cxr_filepath, 
            args=args
        ) 

    y_pred, pos_pred, neg_pred = run_softmax_eval(model, loader, cxr_labels, cxr_pair_template)
    cxr_true_labels_path = osp.join(os.path.abspath('.'), "test_sample_sp.csv")
    # cxr_labels: List[str] = ['Atelectasis','Cardiomegaly', 
                                         # 'Consolidation', 'Edema','Pleural Effusion']
    cxr_labels: List[str] = ['Atelectasis','Cardiomegaly', 
                                          'Consolidation', 'Edema','Pleural Effusion', 'Hilar Enlargement', 'Pneumothorax', 'Pneumonia', 'Vascular Redistribution', 'Heart Insufficiency']
#     cxr_labels: List[str] = ['Atelectasis','Pleural Effusion', 
#                                           'Cardiomegaly','Pneumonia', 'Heart Insufficiency']
    
    test_true = make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

    # compute AUC
    print('test_true', test_true)
    print('y_pred', y_pred)
    sample_max = []
    count = 0
    
    auc = roc_auc_score(test_true, y_pred)
    print('acc', auc)
    acc_total, f1_total = {}, {}
    acc_mean = 0
    f1_mean = 0
    for q in range(len(cxr_labels)): 
        pred = np.concatenate((pos_pred[:, q].reshape(-1,1), neg_pred[:, q].reshape(-1,1)), axis=1)
        pred = np.argmax(pred, axis=1) # get the index of the positive class
        acc = accuracy_score(test_true[:, q], pred)
        f1 = f1_score(test_true[:, q], pred)
        acc_total[cxr_labels[q]] = acc
        f1_total[cxr_labels[q]] = f1
        
        acc_mean += acc
        f1_mean += f1
    
    acc_mean = acc_mean / len(acc_total)
    f1_mean = f1_mean / len(acc_total)
    print('acc', acc_mean)
    print('f1', f1_mean)
    
    print('acc_total: ',  acc_total)
    
    # f1_total = f1_score(test_true, y_pred, average='samples')
    print('f1_total: ', f1_total)
    
        

    