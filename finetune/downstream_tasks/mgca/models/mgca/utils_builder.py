from cgi import test
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision.models import resnet as torch_resnet
import torch.nn.functional as F
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer

from transformers import BertModel
from transformers import AutoConfig
from transformers.models.bert.tokenization_bert import BertTokenizer
import os
import os.path as osp
# raw resnet with cxrbert-genereal
class Language_discriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
    """

    def __init__(self, input_size=512, num_classes=1):
        super(Language_discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, h):
        y = self.layer(h)
        return y

class ResNet_CXRBert(torch.nn.Module):
    def __init__(self, args):
        super(ResNet_CXRBert, self).__init__()
        self.args = args
        
        self.Language_discriminator = Language_discriminator()
        self.encoder = torchvision.models.resnet50(pretrained=False)
        self.encoder.fc = nn.Identity()

        checkpoint = torch.load(self.args.vision_model, map_location=torch.device('cpu'))
        self.encoder.load_state_dict(checkpoint)
        

        self.proj_v = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512, affine=False))

        self.proj_t = nn.Sequential(
            nn.Linear(768, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512, affine=False))
        
        self.proj_t_constrast = nn.Sequential(
            nn.Linear(768, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024, affine=False))

        self.tokenizer = BertTokenizer.from_pretrained(self.args.lm_model,
                                                           do_lower_case=True)
        self.config = AutoConfig.from_pretrained(self.args.lm_model + '/config.json',
                                                     cache_dir=None)
        
        self.lm_model = BertModel.from_pretrained(
                args.lm_model + '/pytorch_model.bin',
                from_tf=bool(".ckpt" in self.args.lm_model + '/pytorch_model.bin'),
                config=self.config,
                cache_dir=None)
        
        print('Loading sperate multi-modal ok for zeroshot ######################')

    def _tokenize(self, text):
        tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                            add_special_tokens=True,
                                                            truncation=True,
                                                            max_length=256,
                                                            padding='max_length',
                                                            return_tensors='pt')

        return tokenizer_output

    def forward(self, img, input_ids, attention_mask):
        img_emb = self.encoder(img)
        # reshape to (b, 2048)
        img_emb = img_emb.view(img_emb.shape[0], img_emb.shape[1])

        # pooler_output: [b, 1, 768]
        text_emb = self.lm_model(input_ids=input_ids,
                                 attention_mask=attention_mask).last_hidden_state

        # project to 512 dim
        proj_img_emb = self.proj_v(img_emb)
        proj_text_emb = self.proj_t(text_emb[:, 0].contiguous())

        return {'img_emb': img_emb,
                'proj_img_emb': proj_img_emb,
                'proj_text_emb': proj_text_emb}

# 2 view resnet with cxrbert-genereal for contrastive loss between 2 views
class ResNet_CXRBert_2view(torch.nn.Module):
    def __init__(self):
        super(ResNet_CXRBert, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)

        self.encoder = resnet
        self.encoder.fc = nn.Identity()

        self.proj_v = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512, affine=False))

        self.proj_t = nn.Sequential(
            nn.Linear(768, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512, affine=False))

        url = 'microsoft/BiomedVLP-CXR-BERT-specialized'
        self.lm_model = AutoModel.from_pretrained(
            url, trust_remote_code=True, revision='main')
        self.tokenizer = AutoTokenizer.from_pretrained(
            url, trust_remote_code=True, revision='main')

    def _tokenize(self, text):
        tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                            add_special_tokens=True,
                                                            truncation=True,
                                                            max_length=128,
                                                            padding='max_length',
                                                            return_tensors='pt')

        return tokenizer_output

    def forward(self, img1, img2, input_ids, attention_mask):
        img_emb_1 = self.encoder(img1)
        img_emb_2 = self.encoder(img2)
        # reshape to (b, 2048)
        img_emb_1 = img_emb.view(img_emb.shape[0], img_emb.shape[1])
        img_emb_2 = img_emb.view(img_emb.shape[0], img_emb.shape[1])

        # pooler_output: [b, 1, 768]
        text_emb = self.lm_model(input_ids=input_ids,
                                 attention_mask=attention_mask).last_hidden_state

        # project to 512 dim
        proj_img_emb = self.proj_v(img_emb_1)
        proj_text_emb = self.proj_t(text_emb[:, 0].contiguous())

        return {'img_emb_1': img_emb_1,
                'proj_img_emb_1': proj_img_emb_1,
                'img_emb_2': img_emb_2,
                'proj_img_emb_2': proj_img_emb_2,
                'proj_text_emb': proj_text_emb}


# simple projection head
class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size),
            nn.BatchNorm1d(projection_size, affine=False))

    def forward(self, x):
        return self.net(x)