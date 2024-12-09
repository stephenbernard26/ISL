"""
Author: Sandeep Kumar Suresh

This code is used for counting the number of non-blur frames in the test-dataset file
which is present in the original model prediction csv file

"""

import pandas as pd
import csv

if __name__ =='__main__':
    
    test_dataset_csv = '/4TBHD/ISL/CodeBase/Test_Dataset_Prediction/right_hand_position_along_body.csv'
    model_pred_csv = '/4TBHD/ISL/CodeBase/model_prediction/right_elbow_orientation.csv'

    df_test = pd.read_csv(test_dataset_csv)
    df_model = pd.read_csv(model_pred_csv)
    df_model = df_model.iloc[13359:]


    # Convert to consistent types, for example, strings
    model_data = list(df_model['Input'].astype(str).str.strip())
    test_data = list(df_test['Input'].astype(str).str.strip())
    

    non_blur_frames = []

    count = 0
    for ele in test_data:
        # print("ele",(str(ele)))
        if str(ele) in model_data:
            # print("yes")
            count += 1

            non_blur_frames.append(ele)

    test_dir = '../Dataset/Test'

    with open(f'{test_dir}/Multiple_person_4715.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        
        
        for a in non_blur_frames:
            writer.writerow([a])
    

    print(f"Number of elements from model_data present in test_data: {count}")
    # print("list",model_data[0:10])


