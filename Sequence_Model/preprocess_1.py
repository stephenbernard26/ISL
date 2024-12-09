"""
Author : Sandeep Kumar Suresh
         EE23S059

"""

import pandas as pd
import pickle
import os


def create_pickle_file(df,file_path,filename='seq_dataset.pkl'):

    folder_dict = {}
    prev_subfolder = None

    for path in df:
        parts = path.split('/')
        current_subfolder = parts[8]

        if current_subfolder in folder_dict:
            folder_dict[current_subfolder].append(path)
        else:
            folder_dict[current_subfolder] = [path]

        prev_subfolder = current_subfolder

    with open(f'{file_path}/{filename}', 'wb') as file: 
        
        pickle.dump(folder_dict, file) 
    



if __name__ == '__main__':

    # Here the dataset path is the excel which contains all the frames excluding the blur frames
    path_to_dataset = '/4TBHD/ISL/CodeBase/10k_manual_annotation/right_elbow_orientation.csv'

    df = pd.read_csv(path_to_dataset)

    df = list(df['Input'].astype(str).str.strip())

    pickle_file_save_dir = '/4TBHD/ISL/CodeBase'

    if not os.path.exists(f'{pickle_file_save_dir}/seq_dataset.pkl'):
        create_pickle_file(df,pickle_file_save_dir)
    

    file_path = f'{pickle_file_save_dir}/seq_dataset.pkl'
    if os.path.getsize(file_path) > 0:
        with open(file_path, 'rb') as f:
            classification_dict = pickle.load(f)
    else:
        print("The file is empty.")
    
    print(classification_dict["stephen_hello_6"])