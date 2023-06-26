import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import os
import os.path as osp
from transformers import BertModel
from transformers import AutoConfig
from transformers.models.bert.tokenization_bert import BertTokenizer
import random
import pandas as pd
from tqdm import  tqdm
import torch
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib.ticker import FuncFormatter

if __name__ == '__main__':
    
    sns.set(style='whitegrid')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(44)
    np.random.seed(44)


    # 1. 载入预训练的BERT模型和分词器
    model_name = osp.join(os.path.abspath('.'), "lm_model/")
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

    config = AutoConfig.from_pretrained(model_name + '/config.json',
                                                     cache_dir=None)

    lm_model = BertModel.from_pretrained(
                model_name + '/pytorch_model.bin',
                from_tf=bool(".ckpt" in model_name + '/pytorch_model.bin'),
                config=config,
                cache_dir=None).to('cuda')

    # 2. 准备文本数据 #########################
    sp_csv = osp.join(os.path.abspath('.'), "texts/PDC_cleaned.csv")
    sp_csv = pd.read_csv(sp_csv)
    en_csv = osp.join(os.path.abspath('.'), "texts/200k_find_imp.csv")
    en_csv = pd.read_csv(en_csv)
    
    # En:
    findings = list(en_csv['findings'])
    impression = list(en_csv['impression'])
    en_text_data = []
    en_line_len = len(findings)
    
    for i in range(en_line_len):

        try:
            text = {'INDI': 'None', 'FIND': findings[i], 'IMP': (findings[i] + impression[i]).replace('dumb', '')}
            # text = {'INDI': 'None', 'FIND': 'None', 'IMP': impression[i]}

        except:
            text = {'INDI': 'None', 'FIND': 'None', 'IMP': impression[i]}
        en_text = text['IMP']
        en_text_data.append(en_text)  # getting en_text data
        
    # Sp:
    sp_text_data = []
    sp_line_len = len(sp_csv['Report'])
    for i in range(sp_line_len):
        text = {'INDI': 'None', 'FIND': 'None', 'IMP': sp_csv['Report'][i]}
        sp_text_data.append(text['IMP'])

    # random sample 1000
    
    sample_sp = random.sample(sp_text_data, 250)
    sample_en = random.sample(en_text_data, 250)
    texts = sample_sp + sample_en


    ############################################################
    
    sp = ['sp' for i in range(1000)]
    en = ['en' for i in range(1000)]
   
    categories = sp + en
    
    category_colors = {'sp': 'deepskyblue', 'en': 'orange'}

    # 3. 对文本进行编码并提取文本嵌入
    embeddings = []
   
    for text in tqdm(texts):
        inputs = tokenizer(text, 
                           add_special_tokens=True,
                           truncation=True,
                           max_length=256,
                           padding='longest',
                           return_tensors='pt')
        inputs = inputs.to('cuda')
        
        outputs = lm_model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].detach().cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)

    # calculate the similarity matrix
    # 计算相似度矩阵
    similarity_matrix = np.inner(embeddings, embeddings)

    # 绘制相似度矩阵可视化图
    fig, ax  = plt.subplots()
    im = ax.imshow(similarity_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, format=FuncFormatter(lambda x, _: x/100))
#     plt.xticks(np.arange(len(texts)), texts, rotation=45)
#     plt.yticks(np.arange(len(texts)), texts)
    plt.title('Similarity Matrix for Medical Reports')
    save_path = osp.join(os.path.abspath('.'), "save_imgs/sim.svg")
    title ='w/o CTR'
#     ax.set_title(title, fontsize = 25)
#     ax.title.set_y(1)

    plt.savefig(save_path, bbox_inches = 'tight')
    plt.show()