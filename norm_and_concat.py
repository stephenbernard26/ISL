"""
Author: Sandeep Kumar Suresh
        EE23S059

This Code reads the inference csv file and upsamples the features. 

The upsampled features are stored in a csv file named 'Upsampled_file.csv' --> tmp file



TODO

Make the code generalizable to multiple csv file input



"""
import random
from collections import Counter
import pandas as pd
import csv
import os
import numpy as np



np.random.seed(42)

def upsample_data(labels, paths,default_max_count = 3000):

    """
    Upsamples a dataset to balance label distribution by increasing the number 
    of instances in underrepresented classes to match the count of the most 
    frequent class.

    Parameters:
    ----------
    labels : list
        A list of labels corresponding to different categories/classes.
        
    paths : list
        A list of file paths or data paths corresponding to the labels.
    
    Returns:
    -------
    upsampled_labels : list
        A list of labels after upsampling, where each label category has been
        balanced to match the frequency of the most common label.
        
    upsampled_paths : list
        A list of paths corresponding to the upsampled labels.
    
    Notes:
    -----
    - The function ensures that the number of samples in each category equals 
      the maximum count by randomly duplicating entries for underrepresented 
      classes.
    - Labels and paths are zipped together and maintained in their respective 
      order during upsampling.
    """

    count_dict = Counter(labels)
    
    max_count = max(count_dict.values())

    class_count = []
    print(max_count)
    for element, count in count_dict.items():
        print(f'{element}: {count}')    
        class_count.append(count)
    
    print("class count", class_count)
    # Calculating Coefficient of Variation
    mean = np.mean(class_count)
    std_dev = np.std(class_count)
    cv = std_dev / mean
    print("cv",cv)


    # Overwriting the max_count value if CV is greater than o.5
    if cv > 0.5:
        max_count = default_max_count
    
    
    category_data = {}
    for label, path in zip(labels, paths):
        if label not in category_data:
            category_data[label] = {'labels': [], 'paths': []}
        category_data[label]['labels'].append(label)
        category_data[label]['paths'].append(path)
    
    upsampled_labels = []
    upsampled_paths = []
    
    for category, data in category_data.items():
        current_count = len(data['labels'])
        print("current count",current_count)

        label_path_pairs = list(zip(data['labels'], data['paths']))

        if current_count < max_count:
            additional_needed = max_count - current_count

            additional_pairs = random.choices(label_path_pairs, k=additional_needed)

            additional_labels, additional_paths = zip(*additional_pairs)

            upsampled_labels.extend(data['labels'] + list(additional_labels))
            upsampled_paths.extend(data['paths'] + list(additional_paths))

        elif current_count == max_count:

        #     additional_pairs = random.choices(label_path_pairs, k=max_count)

        #     additional_labels, additional_paths = zip(*additional_pairs)

            upsampled_labels.extend(data['labels'] ) 
            upsampled_paths.extend(data['paths'] )  



        else:

            selected_pairs = random.sample(label_path_pairs, k=max_count)

            selected_labels, selected_paths = zip(*selected_pairs)

            upsampled_labels.extend(list(selected_labels))
            upsampled_paths.extend(list(selected_paths))


    return upsampled_labels, upsampled_paths

def write_to_csv(list_a, list_b, filename="/4TBHD/ISL/CodeBase/tmp/Upsampled_file.csv"):
    if len(list_a) != len(list_b):
        raise ValueError("Both lists must have the same length")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        
        for a, b in zip(list_a, list_b):
            writer.writerow([a, b])

def concatenate_labels(path_to_upsampled_csv , path_to_gold_label_csv,default_path = './Normalized_Concat_Data'):

    # Creating a directory to save the concatinated Normalized Data

    os.makedirs(default_path,exist_ok=True)

    gold_label = pd.read_csv(path_to_gold_label_csv)
    upsampled_label = pd.read_csv(path_to_upsampled_csv)

    new_filename = (path_to_gold_label_csv.split('/')[-1]).split('.')[0]


    # Concatenating column 1 of A to column 1 of B and column 2 of A to column 2 of B
    concat_col1 = pd.concat([gold_label.iloc[:, 0], upsampled_label.iloc[:, 0]], ignore_index=True)
    concat_col2 = pd.concat([gold_label.iloc[:, 1], upsampled_label.iloc[:, 1]], ignore_index=True)

    # Creating a new dataframe with the concatenated columns
    concat_label = pd.DataFrame({
        gold_label.columns[0]: concat_col1,
        gold_label.columns[1]: concat_col2
    })

    # concat_label = gold_label._append(upsampled_label)
    # concat_label = pd.concat([gold_label, upsampled_label])

    concat_label.to_csv(f"{default_path}/{new_filename}.csv", index=False)



        



if __name__ == '__main__':

    # Change name of file
    filename = 'hand_synchronization'
    # path_to_excel = f'/4TBHD/ISL/CodeBase/Test_Dataset_Corrections/{filename}.csv'
    # path_to_excel = f'/4TBHD/ISL/CodeBase/Dataset/non_normalized_concat/v2_post4.7k/{filename}.csv'
    path_to_excel = f'/4TBHD/ISL/CodeBase/Dataset/two_hand_dataset/non_blur_data/{filename}.csv'

    # path_to_gold_label_csv = f'/4TBHD/ISL/CodeBase/Dataset/Train/v1_10k_normalized/new_{filename}_concat.csv'
    # path_to_gold_label_csv = '/4TBHD/ISL/CodeBase/tmp/tmp.csv'



    no_of_rows = 1000000


    df = pd.read_csv(path_to_excel)
    # df = pd.read_csv(path_to_gold_label_csv)


    df_head = df.head(no_of_rows)

    path,corrections = [],[]

    for index, row in df_head.iterrows():
        # input_value = row['Input'].replace('onlyface','keypoints').replace('jpg','npy')
        # prediction_value = row['Corrections']
        input_value = row['image_path'].replace('onlyface','keypoints').replace('jpg','npy')
        prediction_value = row[f'{filename}']

        
        path.append(input_value)
        corrections.append(prediction_value)

    print(len(path))
    
    # print(np.unique(corrections))
        # # Append modified values to the results list
        # results.append((input_value, prediction_value))



    # path = list(df_head['Input'])

    # corrections = list(df_head['Corrections'])

    # print(len(corrections))
    # print(len(path))



    upsampled_labels, upsampled_paths = upsample_data(corrections, path,default_max_count=1000)

    # Checking results
    print("Upsampled Labels:", len(upsampled_labels))
    print("Upsampled Paths:", len(upsampled_paths))

    # write_to_csv(upsampled_paths,upsampled_labels)
    write_to_csv(upsampled_paths,upsampled_labels,filename=f'/4TBHD/ISL/CodeBase/Dataset/two_hand_dataset/normalised_data/{filename}.csv')




    # concatenate_labels(path_to_upsampled_csv='/4TBHD/ISL/CodeBase/tmp/Upsampled_file.csv',path_to_gold_label_csv=path_to_gold_label_csv)


    

    count_dict = Counter(upsampled_labels)
    
    max_count = max(count_dict.values())

    class_count = []
    print(max_count)
    for element, count in count_dict.items():
        print(f'{element}: {count}')    
        class_count.append(count)
    
    print("class count", class_count)
    # Calculating Coefficient of Variation
    mean = np.mean(class_count)
    std_dev = np.std(class_count)
    cv = std_dev / mean
    print("cv",cv)

    # for index, value in df_head['Corrections'].items():
    #     if pd.isna(value):
    #         print(f"NaN found at index {index}")

    for index, value in df_head[filename].items():
        if pd.isna(value):
            print(f"NaN found at index {index}")