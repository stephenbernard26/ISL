"""
Author : Sandeep Kumar Suresh

This code is to extract reliable non blur frames from the existing csv file

"""

import pandas as pd
import csv
import os

def calculate_accuracy(pred, actual):
    correct_predictions = sum(p == a for p, a in zip(pred, actual))
    accuracy = correct_predictions / len(actual)
    return accuracy

if __name__ =='__main__':
    
    test_dataset_csv = '/4TBHD/ISL/CodeBase/Test_Dataset_Corrections/right_palm_position.csv'
    model_pred_csv = '/4TBHD/ISL/CodeBase/Dataset/modelv1_inference/right_elbow_orientation.csv'

    # model_4714 = '/4TBHD/ISL/CodeBase/Test_Dataset_Prediction/right_fingers_joined_4714.csv'

    df_test = pd.read_csv(test_dataset_csv)
    df_model = pd.read_csv(model_pred_csv)
    # df_4714 = pd.read_csv(model_4714)

    df_model = df_model.iloc[13359:]


    # Convert to consistent types, for example, strings
    model_data = list(df_model['Input'].astype(str).str.strip())
    test_data = list(df_test['Input'].astype(str).str.strip())
    corrections_value = list(df_test['Corrections'].astype(str).str.strip())
    # test_4714_ip= list(df_4714['Input'].astype(str).str.strip())
    # test_4714_pred= list(df_4714['Prediction'].astype(str).str.strip())
    

    non_blur_frames,correction = [],[]

    count = 0
    for ele ,val in zip(test_data,corrections_value):
        # print("ele",(str(ele)))
        if str(ele) in model_data:
            # print(ele)

            # print("yes")
            count += 1

            non_blur_frames.append(ele)
            correction.append(val)

            # break
    
    # dictionary = dict(zip(non_blur_frames,correction))

    # print(dictionary)

    # Specify the CSV file name
    csv_file = "/4TBHD/ISL/CodeBase/tmp/right_palm_position_4714.csv"

    # Writing dictionary to CSV
    with open(csv_file, mode='w', newline='') as file:
        # Create a writer object
        writer = csv.writer(file)

        # Write the header
        writer.writerow(["Input", "Corrections"])

        for item1, item2 in zip(non_blur_frames, correction):
            writer.writerow([item1, item2])

    print(f"Data written to {csv_file} successfully.")






    # print(non_blur_frames[:10])
    # corrections_list=[]
    # for j in test_4714_ip:
    #     corrections_list.append(dictionary[j])

    # print(len(corrections_list))

    # x = calculate_accuracy(test_4714_pred,corrections_list)
    # print(x)