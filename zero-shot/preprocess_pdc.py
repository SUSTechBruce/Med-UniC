import pandas as pd
import os.path as osp
import os 

if __name__ =='__main__':
    
#     total_csv = osp.join(os.path.abspath('.'), "PDC_cleaned.csv")
#     total_df = pd.read_csv(total_csv)
    
#     total_df = total_df.loc[total_df['Labels'].str.contains('atelectasis|consolidation|pleural effusion|cardiomegaly|pneumothorax|vascular redistribution|heart insufficiency|pneumonia|hilar enlargement|edema|lesion|normal', na=False)]
    
#     df = total_df
#     df_file = osp.join(os.path.abspath('.'), "PDC_zeroshot_new.csv")
#     df.to_csv(df_file, index=True, index_label ='img_id')
#     print(df)
    
    

    ######### filtering physician data ############################# 
    filter_csv = osp.join(os.path.abspath('.'), "PDC_zeroshot_new.csv")
    df =  pd.read_csv(filter_csv)
#     new_df = df.loc[(df['MethodLabel'] =="Physician")][:1000]
    new_df  = df

    intial = [0 for i in range(len(new_df['Labels']))]
    
    new_pdc_labels = {'img_id': [], 'Atelectasis': [], 'Consolidation': [] , 'Edema': [],  'Pleural Effusion': [],  'Cardiomegaly': [],  'Pneumothorax': []
                     , 'Vascular Redistribution': [], 'Heart Insufficiency': [],  'Pneumonia': [],  'Hilar Enlargement': [], 'Lesion': [], 'Normal':[]}
    for i in range(len(new_df['Labels'])):
        new_pdc_labels['img_id'].append(new_df['img_id'][i])
        
        if "'atelectasis" in new_df['Labels'][i] :

            new_pdc_labels['Atelectasis'].append(1)
        else:
            new_pdc_labels['Atelectasis'].append(0)
            
        if "consolidation" in new_df['Labels'][i]:

            new_pdc_labels['Consolidation'].append(1)
        else:
            new_pdc_labels['Consolidation'].append(0)
            
        if "edema" in new_df['Labels'][i]:

            new_pdc_labels['Edema'].append(1)
        else:
            new_pdc_labels['Edema'].append(0)
            
        if "pleural effusion" in new_df['Labels'][i]:
            

            new_pdc_labels['Pleural Effusion'].append(1)
        else:
            new_pdc_labels['Pleural Effusion'].append(0)
        
        if "cardiomegaly" in new_df['Labels'][i]:
  
            new_pdc_labels['Cardiomegaly'].append(1)
        else:
            new_pdc_labels['Cardiomegaly'].append(0)
        
        if "pneumothorax" in new_df['Labels'][i]:

            new_pdc_labels['Pneumothorax'].append(1)
        else:
            new_pdc_labels['Pneumothorax'].append(0)
        
        if "vascular redistribution" in new_df['Labels'][i]:

            new_pdc_labels['Vascular Redistribution'].append(1)
        else:
            new_pdc_labels['Vascular Redistribution'].append(0)
            
        if "heart insufficiency" in new_df['Labels'][i]:

            new_pdc_labels['Heart Insufficiency'].append(1)
        else:
            new_pdc_labels['Heart Insufficiency'].append(0)
            
        if "pneumonia" in new_df['Labels'][i]:

            new_pdc_labels['Pneumonia'].append(1)
        else:
            new_pdc_labels['Pneumonia'].append(0)
            
        if "hilar enlargement" in new_df['Labels'][i]:
    
            new_pdc_labels['Hilar Enlargement'].append(1)
        else:
            new_pdc_labels['Hilar Enlargement'].append(0)
            
        if "lesion" in new_df['Labels'][i]:
    
            new_pdc_labels['Lesion'].append(1)
        else:
            new_pdc_labels['Lesion'].append(0)
            
        if "normal" in new_df['Labels'][i]:
    
            new_pdc_labels['Normal'].append(1)
        else:
            new_pdc_labels['Normal'].append(0)
            
            

            
    new_pdc_labels = pd.DataFrame(new_pdc_labels)
    
#     new_pdc_labels = new_pdc_labels.drop(new_pdc_labels[(new_pdc_labels['Hilar Enlargement']==0) & (new_pdc_labels['Heart Insufficiency']==0)
#                                                        & (new_pdc_labels['Consolidation']==0) & (new_pdc_labels['Pneumonia']==0)
#                                                        & (new_pdc_labels['Pleural Effusion']==0) & (new_pdc_labels['Vascular Redistribution']==0)
#                                                        & (new_pdc_labels['Atelectasis']==0) & (new_pdc_labels['Edema']==0)
#                                                        & (new_pdc_labels['Cardiomegaly']==0) & (new_pdc_labels['Pneumothorax']==0) & (new_pdc_labels['Lesion']==0) & (new_pdc_labels['Normal']==0)].index) 
    
    new_pdc_labels = new_pdc_labels.drop(new_pdc_labels[(new_pdc_labels['Hilar Enlargement'] + new_pdc_labels['Heart Insufficiency'] + new_pdc_labels['Consolidation']
                                                        + new_pdc_labels['Pneumonia'] + new_pdc_labels['Pleural Effusion'] + new_pdc_labels['Vascular Redistribution'] + new_pdc_labels['Atelectasis'] + new_pdc_labels['Edema'] + new_pdc_labels['Cardiomegaly'] + new_pdc_labels['Pneumothorax']) <= 1].index) 
#     new_pdc_labels = new_pdc_labels[:1000]
    print(new_pdc_labels)
    filter_zeroshot = osp.join(os.path.abspath('.'), "final_pdc_labels.csv")
    new_pdc_labels.to_csv(filter_zeroshot, index=False)
    new_pdc_labels = new_pdc_labels.iloc[[1, 2, 3]]
    print(new_pdc_labels)
    
    pdc_sp_csv = osp.join(os.path.abspath('.'), "test_sample_sp.csv")
    pdc_sp = pd.read_csv(pdc_sp_csv)
    pdc_sp = pdc_sp[:1000]
    so_out = osp.join(os.path.abspath('.'), "pdc_zeroshot_sp.csv")
    pdc_sp.to_csv(so_out, index=False)
    
    pdc_en_csv = osp.join(os.path.abspath('.'), "test_sample.csv")
    pdc_en = pd.read_csv(pdc_en_csv)
    pdc_en = pdc_en[:1000]
    en_out = osp.join(os.path.abspath('.'), "pdc_zeroshot_en.csv")
    pdc_en.to_csv(en_out, index=False)
    
    
    
    