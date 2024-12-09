"""
Inference Code for Sequence Classifier

Input: Video path
Output : Sign

"""
import sys
import os
import mediapipe as mp
import yaml
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from model import *
from sequence_classifier import *
from custom_dataloader import *
from utility import *
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(parent_dir)
sys.path.append(parent_dir)
from extract_non_blur_frames_server import *
from preprocess import *

mp_holistic = mp.solutions.holistic  # used to extract full body keypoints


if torch.cuda.is_available():  
    device = "cuda" 
else:  
    device = "cpu"  

print("device",device)


if __name__ == '__main__':

    config = read_config('config.yaml')
    input_dim =  config['inference']["input_dim"]
    hidden_dim =  config['inference']["hidden_dim"]


    # npy_dict = {}
    # npy_dict = defaultdict(list)
    npy_base_dir = "/4TBHD/ISL/CodeBase/Sequence_Model/seq_npy_folder"
    frame_tmp_dir = '/4TBHD/ISL/CodeBase/tmp/frames_tmp'
    npy_tmp_dir = '/4TBHD/ISL/CodeBase/tmp/npy_tmp'
    
    os.makedirs(npy_base_dir,exist_ok=True)
    os.makedirs(frame_tmp_dir,exist_ok=True)
    os.makedirs(npy_tmp_dir,exist_ok=True)

    test_data_path = '/4TBHD/ISL/data_preparation/seq_test_data/Janaghan/beautiful/Janaghan_Beautiful_1.mp4'
    
    # Comment the below code if your dataset is already converted to pickle file
    construct_npy_file(config,npy_tmp_dir,frame_tmp_dir,test_data_path,npy_pickle_path='sv_inference_tmp_npy.pkl')

    file_path = 'sv_inference_tmp_npy.pkl'
    if os.path.getsize(file_path) > 0:
        with open(file_path, 'rb') as f:
            classification_dict = pickle.load(f)
    else:
        print("The file is empty.")

    # # The code below is used to read the pickle file and make the seq_data_pickle file
    dataset_size = "expanded" # Reduced/Expanded
    construct_seq_dictionary(config,classification_dict,dataset_size,save_path='sv_inference_tmp_seq.pkl')

    file_path_2 = 'sv_inference_tmp_seq.pkl'
    if os.path.getsize(file_path_2) > 0:
        with open(file_path_2, 'rb') as f:
            seq_ip_data = pickle.load(f)
    else:
        print("The file is empty.")


    df_seq_ip = pd.DataFrame(seq_ip_data)

    # print(df_seq_ip)

    data = SequenceClassificationDataset(df_seq_ip)
    dataloader = torch.utils.data.DataLoader(data)


    label_mapping = {0: 'beautiful', 1: 'brush', 2: 'bye', 3: 'cook', 4: 'dead', 5: 'go', 6: 'good', 7: 'hello', 8: 'sorry', 9: 'thankyou'}

    seq_model = 'LSTM'
    input_size = 8
    hidden_size = 256
    num_layers = 1
    output_size = 10
    dropout = 0.2
    max_epoch = 50
    learning_rate = 0.0001
    



    model = Classifier(seq_model, input_size, hidden_size, num_layers, output_size ,dropout)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.load_state_dict(torch.load(config['seq_classifier_inference']["checkpoint_path"]))
    model.to(device)

    predictions = []
    true_labels = []
    with torch.no_grad():  # No gradient calculations
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)

                # Forward pass
            pred = model(X)
            predicted_labels = torch.argmax(pred, dim=1)  # Convert logits to class predictions
                
            predictions.extend(predicted_labels.cpu().numpy())
            true_labels.extend(Y.cpu().numpy())

    print(type(predictions))
    #   predictions_list = (predictions.values())
    string_predictions_list = [label_mapping[prediction] for prediction in predictions]
    true_labels_list = [label_mapping[label] for label in true_labels]

    print(string_predictions_list)
    print(true_labels_list)

    acc = calculate_accuracy(string_predictions_list,true_labels_list)

    print(acc)
    print(calculate_classwise_accuracy(string_predictions_list,true_labels_list))
