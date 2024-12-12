import os
import pandas as pd
import csv
import random
from tqdm import tqdm
if __name__ == '__main__':

    path_to_dataset_folder =  '/4TBHD/ISL/data_preparation/test_all/onlyface'

    manual_annotation = '/4TBHD/ISL/CodeBase/10k_manual_annotation/right_elbow_orientation.csv'
    new_norm_10k_data = '/4TBHD/ISL/CodeBase/Dataset/Train/v2_4.7k_normalized/new_right_fingertips_orientation_concat.csv'

    df_manual_annotation = pd.read_csv(manual_annotation)
    df_new_norm_10k_data = pd.read_csv(new_norm_10k_data)

    df_manual_annotation = list(df_manual_annotation['Input'].astype(str).str.strip())
    df_new_norm_10k_data = list(set(list(df_new_norm_10k_data['Image Name'].astype(str).str.strip())))

    print(type(df_new_norm_10k_data))

    folders_to_exclude = ["Hari","Ishan","Test"]

    main_list = []
    for Signer in tqdm(os.listdir(path_to_dataset_folder)):
        if Signer not in folders_to_exclude:
            for subfolders in os.listdir(os.path.join(path_to_dataset_folder,Signer)):
                for subsubfolders in os.listdir(os.path.join(path_to_dataset_folder,Signer,subfolders)):
                    # print(subsubfolders)
                    img_sample = []
                    for img in os.listdir(os.path.join(path_to_dataset_folder,Signer,subfolders,subsubfolders)):
                        if int(img.split('.')[0]) > 20 and int(img.split('.')[0]) < 60 :
                            path_image = os.path.join(path_to_dataset_folder,Signer,subfolders,subsubfolders,img)
                            if (path_image in df_manual_annotation) and (path_image not in df_new_norm_10k_data):
                                # print('hi')
                                img_sample.append(path_image)
                    if len(img_sample) >= 1:
                        main_list.extend(random.sample(img_sample, k=1))
            
                                        
            
    test_dir = '/4TBHD/ISL/CodeBase/Dataset/Test'

    with open(f'{test_dir}/test_dataset_v2.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        
        
        for a in main_list:
            writer.writerow([a])