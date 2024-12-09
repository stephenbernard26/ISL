import os
import pandas as pd
import csv
import random

if __name__ == '__main__':

    path_to_dataset_folder =  '/4TBHD/ISL/data_preparation/test_all/onlyface'

    folders_to_exclude = ['Ishan','Test','Hari']

    main_list = []
    for Signer in os.listdir(path_to_dataset_folder):
        if Signer not in folders_to_exclude:
            for subfolders in os.listdir(os.path.join(path_to_dataset_folder,Signer)):
                for subsubfolders in os.listdir(os.path.join(path_to_dataset_folder,Signer,subfolders)):
                    img_sample = []
                    for img in os.listdir(os.path.join(path_to_dataset_folder,Signer,subfolders,subsubfolders)):
                        if int(img.split('.')[0]) > 20 and int(img.split('.')[0]) < 50 :
                            img_sample.append(os.path.join(path_to_dataset_folder,Signer,subfolders,subsubfolders,img).replace('onlyface','keypoints').replace('jpg','npy'))
                    main_list.extend(random.sample(img_sample,k=3))





    # df = pd.read_csv(path_to_dataset)

    # df_subset = df.iloc[13357:]

    test_dir = '../Dataset/Test'
    # os.makedirs( test_dir, exist_ok=True)

    # test_dataset = []
    # for index, row in df_subset.iterrows():
    #     img_no = (row['Input'].split('/')[-1]).split('.')[0]
    #     if (int(img_no) > 20 and int(img_no) < 50 ):
    #         test_dataset.append(row['Input'])


    with open(f'{test_dir}/test_dataset.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        
        
        for a in main_list:
            writer.writerow([a])
    