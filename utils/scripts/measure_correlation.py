import random
from collections import Counter
import pandas as pd
import csv
import os
import numpy as np
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(parent_dir)
import norm_and_concat



np.random.seed(42)

if __name__ == '__main__':

    path = '/4TBHD/ISL/CodeBase/tmp/non_blur_frames.csv'

    filename = 'right_fingertips_orientation'
    path_to_gold_label_csv = f'/4TBHD/ISL/CodeBase/Normalized_Concat_Data/new_{filename}_concat.csv'


    df = pd.read_csv(path)

    correction_value , path = [],[]

    for index, row in df.iterrows():

        input_value = row['Input']        
        prediction_value = row['Corrections']
        
        correction_value.append(prediction_value)
        path.append(input_value)
    

    upsampled_labels, upsampled_paths = norm_and_concat.upsample_data(correction_value, path ,600)

    # Checking results
    print("Upsampled Labels:", len(upsampled_labels))
    print("Upsampled Paths:", len(upsampled_paths))

    norm_and_concat.write_to_csv(upsampled_paths,upsampled_labels)
    norm_and_concat.concatenate_labels(path_to_upsampled_csv='/4TBHD/ISL/CodeBase/tmp/Upsampled_file.csv',
                                       path_to_gold_label_csv=path_to_gold_label_csv,
                                       default_path = '/4TBHD/ISL/CodeBase/New_Normalized_Concat_Data')
