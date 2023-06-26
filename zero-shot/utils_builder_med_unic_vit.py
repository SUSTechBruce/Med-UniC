import torch
import torch.nn as nn

import torchvision

from transformers import BertModel
from transformers import AutoConfig
from transformers.models.bert.tokenization_bert import BertTokenizer
import torch.nn.functional as F
from torch.nn import  BCELoss
from vits import create_vit

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

### raw resnet with cxrbert-genereal

class ResNet_CXRBert(torch.nn.Module):
    def __init__(self, args):
        super(ResNet_CXRBert, self).__init__()
        self.args = args

        # loading vision encoder
        vision_encoder_name = 'vit'

        if vision_encoder_name == 'vit':
            print('Start to loading vit model')
            vit_grad_ckpt = False
            vit_ckpt_layer = 0
            image_size = 224
            vit_name = 'base'
            self.encoder, vision_width = create_vit(vit_name, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)

            self.feature_dim = vision_width

            vit_cpt = self.args.vision_model
            vit_cpt = torch.load(vit_cpt, map_location=torch.device('cpu'))['model']
            self.encoder.load_state_dict(vit_cpt)
        else:
            print('Start to loading imgnet50 %%%%%%%%%%')
            self.encoder = torchvision.models.resnet50(pretrained=False)
            checkpoint = torch.load(self.args.vision_model_path, map_location=torch.device('cpu'))
            self.encoder.load_state_dict(checkpoint)
            self.encoder.fc = nn.Identity()

        if vision_encoder_name == 'vit':
            self.proj_v = nn.Sequential(
                nn.Linear(self.feature_dim, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 512),
                nn.BatchNorm1d(512, affine=False))
        else:
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


        ################## loading the cxr_bert_model ##############################################



        self.tokenizer = BertTokenizer.from_pretrained(self.args.lm_model,
                                                           do_lower_case=True)
        self.config = AutoConfig.from_pretrained(self.args.lm_model + '/config.json',
                                                     cache_dir=None)
        
        self.lm_model = BertModel.from_pretrained(
                args.lm_model + '/pytorch_model.bin',
                from_tf=bool(".ckpt" in self.args.lm_model + '/pytorch_model.bin'),
                config=self.config,
                cache_dir=None)
        
        self.lm_model.resize_token_embeddings(len(self.tokenizer))


    def freeze_model_layer(self, model, freeze_layers):

        layer_map = {'0': 'layer.0.', '1': 'layer.1.', '2': 'layer.2.', '3': 'layer.3.', '4': 'layer.4.',
                     '5': 'layer.5.',
                     '6': 'layer.6.', '7': 'layer.7.', '8': 'layer.8.', '9': 'layer.9.', '10': 'layer.10.',
                     '11': 'layer.11.'}

        layer_nums = len(layer_map)
        unfree_layers = layer_nums - freeze_layers
        layer_idxs = [layer_map[str(i)] for i in range(layer_nums - unfree_layers, layer_nums)]

        for name, par in model.named_parameters():
            par.requires_grad = False

        for name, par in model.named_parameters():
            for layer in layer_idxs:
                if layer in name:
                    par.requires_grad = True

    def freeze_model_layer_new(self, model, freeze_layers):
        print('USING NEW FREEZE')

        layer_map = {'0': 'layer.0.', '1': 'layer.1.', '2': 'layer.2.', '3': 'layer.3.', '4': 'layer.4.',
                     '5': 'layer.5.',
                     '6': 'layer.6.', '7': 'layer.7.', '8': 'layer.8.', '9': 'layer.9.', '10': 'layer.10.',
                     '11': 'layer.11.'}

        layer_idxs = [layer_map[str(i)] for i in range(0, freeze_layers)]

        for name, par in model.named_parameters():
            for layer in layer_idxs:
                if layer in name:
                    par.requires_grad = False



    def text_discim_loss(self, x, domain_y):

        Loss_func = BCELoss()
        loss_D = Loss_func(x, domain_y)
        return loss_D

    def covar_loss_feature(self, x1, x2):
        def off_diagonal(x):
            # return a flattened view of the off-diagonal elements of a square matrix
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        logits = torch.mm(x1.T, x2).to(self.args.device) # dim batch, batch dim

        logits.div_(self.args.batch_size)
        on_diag = torch.diagonal(logits).add(-1).pow(2).sum()
        off_diag = off_diagonal(logits).pow(2).sum()
        loss = on_diag + 0.0051 * off_diag
        return loss / 2. / x1.shape[1]

    def covar_loss_instance(self, x1, x2):
        def off_diagonal(x):
            # return a flattened view of the off-diagonal elements of a square matrix
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        logits = torch.mm(x1, x2.T).to(self.args.device) # batch, dim * dim, batch

        logits.div_(self.args.batch_size)
        on_diag = torch.diagonal(logits).add(-1).pow(2).sum()
        off_diag = off_diagonal(logits).pow(2).sum()
        loss = on_diag + 0.0051 * off_diag
        return loss / 2. / x1.shape[0]  /50.

    def clip_loss(self, x, y):

        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        sim = torch.einsum('i d, j d -> i j', x, y) / 0.07  # set tau to 0.7

        labels = torch.arange(x.shape[0]).type_as(sim).long().to(self.args.device)

        loss_t = F.cross_entropy(sim, labels)
        loss_i = F.cross_entropy(sim.T, labels)

        i2t_acc1, i2t_acc5 = self.precision_at_k(
            sim, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = self.precision_at_k(
            sim.T, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.

        return (loss_t + loss_i) / 2.0, acc1, acc5


    def precision_at_k(self, output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''

        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)

            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def _tokenize(self, text):

        tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                            add_special_tokens=True,
                                                            truncation=True,
                                                            max_length=256,
                                                            padding='longest',
                                                            return_tensors='pt')
        return tokenizer_output

    def forward(self, img1, img2, domain_true, input_ids, attention_mask, unified_train=True, text_aug=0):

        img_emb_1 = self.encoder(img1)
        if self.args.vision_encoder_name == 'vit':
            img_emb_1 = img_emb_1[:, 0].contiguous()
        img_emb_1 = img_emb_1.view(img_emb_1.shape[0], img_emb_1.shape[1])

        img_emb_2 = self.encoder(img2)
        # reshape to (b, 2048)
        if self.args.vision_encoder_name == 'vit':
            img_emb_2 = img_emb_2[:, 0].contiguous()
        img_emb_2 = img_emb_2.view(img_emb_2.shape[0], img_emb_2.shape[1])


        # pooler_output: [b,L, 768]
        text_emb_1 = self.lm_model(input_ids=input_ids,
                                 attention_mask=attention_mask).last_hidden_state


        text_emb_2 = self.lm_model(input_ids=input_ids,
                                   attention_mask=attention_mask).last_hidden_state

        project_t_contras_emb_2 = self.proj_t_constrast(text_emb_2[:, 0].contiguous())


        # project to 512 dim
        proj_img_emb_1 = self.proj_v(img_emb_1)
        proj_img_emb_2 = self.proj_v(img_emb_2)

        proj_text_emb = self.proj_t(text_emb_1[:, 0].contiguous())

        clip_loss, ti_acc1, ti_acc5 = self.clip_loss(proj_img_emb_1, proj_text_emb) # using slip loss for ablation stud

        # project text embedding to contrastive embedding

        project_t_contras_emb_1 = self.proj_t_constrast(text_emb_1[:, 0].contiguous())

        if unified_train:
            slip_loss, ii_acc1, ii_acc5 = self.clip_loss(proj_img_emb_1, proj_img_emb_2)

            if text_aug:

                text_instance_loss = self.covar_loss_instance(project_t_contras_emb_1, project_t_contras_emb_2)
                text_feature_loss = self.covar_loss_feature(project_t_contras_emb_1, project_t_contras_emb_2)
                text_contras_loss = text_instance_loss + text_feature_loss

                return {
                    'clip_loss': clip_loss,
                    'slip_loss': slip_loss,
                    'text_d_loss': text_contras_loss,
                    'text_instance_loss':text_instance_loss,
                    'text_feature_loss': text_feature_loss,
                    'ti_acc1': ti_acc1,
                    'ti_acc5': ti_acc5,
                    'ii_acc1': ii_acc1,
                    'ii_acc5': ii_acc5
                }


            else:


                text_contras_loss = self.covar_loss_feature(project_t_contras_emb_1, project_t_contras_emb_2)
                return {
                        'clip_loss': clip_loss,
                        'slip_loss': slip_loss,
                        'text_d_loss': text_contras_loss,
                        'ti_acc1': ti_acc1,
                        'ti_acc5': ti_acc5,
                        'ii_acc1': ii_acc1,
                        'ii_acc5': ii_acc5
                }
        else:
            return {
                    'clip_loss': clip_loss,
                    'ti_acc1': ti_acc1,
                    'ti_acc5': ti_acc5,
            }



### simple projection head
class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

