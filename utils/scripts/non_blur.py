"""
This codes takes the non-blur csv file and gives rejects the blur frames which have
been crept in.

Later be intergrated into the "CodeBase/extract_non_blur_frames_server.py"

"""

import os
import numpy as np
import pandas as pd
import csv

if __name__ == '__main__':

    path_to_csv = '../non_blur_frames.csv'

    new_file_path = "./updated_non_blur_frames.csv"

    df = pd.read_csv(path_to_csv, header=None)

    npy_file_paths = df[0].tolist() 

    for path in npy_file_paths:

        npy_load = np.load(path)
        # print(npy_load)
        if np.sum(npy_load[-4]) != 0.0:

            with open(new_file_path, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([path])


