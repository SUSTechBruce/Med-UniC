import random

import torch
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
from randaugment import RandomAugment

class PDC_img_text_embed_dataset(Dataset):

    def __init__(self, image_data, text_data, database='MIMIC', transform=None, train_test=None):
        self.img_data = image_data

        self.text_data = text_data
        self.mode = train_test
        self.database = database
        self.transform = transform
        self.visited = np.zeros(len(text_data))
        assert len(image_data) == len(text_data)

    def __len__(self):
        return (self.img_data.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = self.text_data[idx]

        image = self.img_data[idx]
        image = Image.fromarray(image).convert("RGB")

        # get raw text
        sample = {'raw_text': text, 'text_label': 0}

        if self.transform:
            # for 2 branch contrastive vision model (not useful for CLIP)
            if self.mode == 'train':

                sample['image1'] = self.transform[0](image)
                sample['image2'] = self.transform[1](image)

            elif self.mode == 'test':
                sample['val_image'] = self.transform(image)

        return sample


class MIMIC_img_text_embed_dataset(Dataset):
    def __init__(self, image_data, text_data, database='MIMIC', transform=None, train_test=None):
        self.img_data = image_data

        self.text_data = text_data
        self.mode = train_test
        self.database = database
        self.transform = transform
        self.visited = np.zeros(len(text_data))
        assert len(image_data) == len(text_data)

    def __len__(self):
        return (self.img_data.shape[0])

    # this is dumb part, not using
    def __get_num_class__(self):
        if self.database == 'MIMIC':
            num_class = 13
        return num_class

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.visited[idx] == 1:
            return None

        # get image
        image = self.img_data[idx]

        image = Image.fromarray(image).convert("RGB")

        text = self.text_data[idx]
        sample = {'raw_text': text, 'text_label': 1}


        if self.transform:
            # for 2 branch contrastive vision model (not useful for CLIP)
            if self.mode == 'train':

                sample['image1'] = self.transform[0](image)
                sample['image2'] = self.transform[1](image)

            elif self.mode == 'test':
                sample['val_image'] = self.transform(image)

        return sample


class I_T_emb_dataset:

    def __init__(self, image_path, csv_path, database='MIMIC'):
        self.image_path = image_path
        self.csv_path = csv_path
        self.database = database
        self.random_aug = False

    def get_dataset(self, train_test):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

        if train_test == 'train':
            print('Apply Train-stage Transform!')
            Transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(224),
                normalize
            ])


            Transforms_super = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(224),
                transforms.RandomRotation(degrees=(0, 90)),
                transforms.RandomGrayscale(p=0.5),
                # transforms.RandomPerspective(distortion_scale=0.5,
                #                                 p=0.5,
                #                                 interpolation=3),
                transforms.RandomAffine(degrees=0,
                                        translate=(0.1, 0.1),
                                        scale=(0.9, 1.1),
                                        shear=10,
                                        resample=False,
                                        fillcolor=0),
                transforms.RandomAutocontrast(p=0.5),
                normalize
            ])


            if self.random_aug:

                print('Using random augmentation to preprocess images')
                Transforms_super = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    RandomAugment(2, 7, isPIL=True,
                                  augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
                    transforms.ToTensor(),
                    normalize,
                ])

            Trans = [Transforms, Transforms_super]


        else:
            Trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(224),
                normalize
            ])
            print('Apply Test-stage Transform!')

        print('####################### Loading English data #################### ', )
        en_img_data = np.load(self.image_path['en_img_path'], allow_pickle=True, mmap_mode='r')
        en_csv = pd.read_csv(self.csv_path['en_text_path'])

        print('####################### Loading Spanish data  #################### ')
        sp_img_data = np.load(self.image_path['sp_img_path'], allow_pickle=True, mmap_mode='r')
        sp_csv = pd.read_csv(self.csv_path['sp_text_path'])


        findings = list(en_csv['findings'])
        impression = list(en_csv['impression'])

        en_text_data = []
        en_line_len = len(findings)

        count = 0

        for i in range(en_line_len):

            try:
                text = {'INDI': 'None', 'FIND': findings[i], 'IMP': (findings[i] + impression[i]).replace('dumb', '')}
                # text = {'INDI': 'None', 'FIND': 'None', 'IMP': impression[i]}

            except:
                text = {'INDI': 'None', 'FIND': 'None', 'IMP': impression[i]}
                print('text', text)
                count += 1

            en_text_data.append(text)

        assert len(en_text_data) == en_img_data.shape[0]

        ### Get the text sp data

        sp_text_data = []
        sp_line_len = len(sp_csv['Report'])
        for i in range(sp_line_len):
            text = {'INDI': 'None', 'FIND': 'None', 'IMP': sp_csv['Report'][i]}
            sp_text_data.append(text)

        assert len(sp_text_data) == sp_img_data.shape[0]

        # sp_new = torch.tensor(sp_img_data.astype(np.float32)).squeeze()  # 224 224
        # print('Squeeze successfully', sp_new.shape)

        en_dataset = MIMIC_img_text_embed_dataset(image_data=en_img_data, text_data=en_text_data,
                                                  database=self.database, transform=Trans, train_test=train_test)

        print('############ Loading en_dataset successfully ######################')
        sp_dataset = PDC_img_text_embed_dataset(image_data=sp_img_data, text_data=sp_text_data,
                                                database=self.database, transform=Trans, train_test=train_test)
        unified_dataset = ConcatDataset([en_dataset, sp_dataset])

        print('Just loading EN dataset ########################## ')

        return unified_dataset
