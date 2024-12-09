"""
Author : Sandeep Kumar Suresh
        EE23S059

The below code takes existing csv file and makes extracts non-blur frames from them.

"""

import pandas as pd
import csv
import numpy as np

if __name__ =='__main__':


    # print(df['file_path'])
    # print(df['left_fingers_closeness_to_face'])
    feature = 'hand_synchronization'
    path,value = [],[]

    # csv_data = f'/4TBHD/ISL/data_preparation/left_hand_features/{feature}.csv'
    csv_data = f'/4TBHD/ISL/data_preparation/two_hand_features/{feature}.csv'

    df = pd.read_csv(csv_data)
    print(df)

    for i,row in df.iterrows():
        path.append(row['file_path'])
        value.append(row[feature])
    


    for file_path,feat_value in zip(path,value):
        print(file_path)
        npy_load = np.load(file_path.replace('onlyface','keypoints').replace('jpg','npy'))
        if np.sum(npy_load[-4]) and np.sum(npy_load[32:35]) != 0.0:                                    
            with open(f"/4TBHD/ISL/CodeBase/Dataset/two_hand_dataset/non_blur_data/{feature}.csv", 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                # csvwriter.writerow(["file_path", feature])
                csvwriter.writerow([file_path,feat_value])

    print(len(df))
