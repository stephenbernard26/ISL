"""
Author: Sandeep Kumar Suresh
        EE23S059
        

This file contains all the utilty fuction 

"""

import torch
import os
import pickle
from model import *
from sequence_classifier import *
from custom_dataloader import *
import yaml
import numpy as np
import sys
import mediapipe as mp
from collections import Counter
# from tqdm import tqdm
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(parent_dir)
sys.path.append(parent_dir)
from extract_non_blur_frames_server import *
from preprocess import *
from sklearn.utils import resample
import pandas as pd
from shift_orgin import *
from load_model_gpu import *
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import defaultdict
if torch.cuda.is_available():  
    device = "cuda" 
else:  
    device = "cpu"  

# device = torch.device('cpu')

torch.set_default_tensor_type(torch.cuda.FloatTensor)

def calculate_model_classwise_accuracy(model, loss_fn, loader, device, num_classes):
    model.eval()  # Evaluation mode
    correct = {i: 0 for i in range(num_classes)}
    total = {i: 0 for i in range(num_classes)}
    overall_correct, overall_total, loss_val = 0, 0, 0
    
    with torch.no_grad():  # Disable gradient computation
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, Y)
            loss_val += loss.item()
            _, predicted = torch.max(outputs, 1)  # Get the class with max score
            
            for label, prediction in zip(Y, predicted):
                if label == prediction:
                    correct[label.item()] += 1
                total[label.item()] += 1
            
            overall_total += Y.size(0)
            overall_correct += (predicted == Y).sum().item()
    
    overall_acc = 100 * overall_correct / overall_total
    loss_val /= len(loader)
    
    print("Overall Accuracy: {:.2f}%".format(overall_acc))
    print("Loss: {:.4f}".format(loss_val))
    
    print("\nClass-wise Accuracy:")
    for class_id in range(num_classes):
        if total[class_id] > 0:
            class_acc = 100 * correct[class_id] / total[class_id]
            print(f"Class {class_id}: {class_acc:.2f}%")
        else:
            print(f"Class {class_id}: No samples")



def calculate_model_accuracy(model,loss_fn, loader,device):
    model.eval()  # Evaluation mode
    correct, total,loss_val = 0, 0 , 0
    with torch.no_grad():  # Disable gradient computation
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs,Y)
            loss_val += loss.item()
            _, predicted = torch.max(outputs, 1)  # Get the class with max score
            total += Y.size(0)
            correct += (predicted == Y).sum().item() 
    acc = 100 * correct / total
    loss_val /= len(loader)

    return acc , loss_val

def calculate_accuracy(predictions, labels):
    """Calculate the accuracy between two lists."""
    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = (correct / len(labels)) * 100  # Convert to percentage
    return accuracy

def calculate_classwise_accuracy(pred, actual):
    class_wise_correct = Counter()
    class_wise_total = Counter()

    for p, a in zip(pred, actual):
        class_wise_total[a] += 1  # Count total instances of each class
        if p == a:
            class_wise_correct[a] += 1  # Count correct predictions for each class

    classwise_accuracy = {cls: class_wise_correct[cls] / class_wise_total[cls]
                          for cls in class_wise_total}
    return classwise_accuracy



def extract_npy_files(config,frame,npy_save_dir,frame_count):

    mp_holistic = mp.solutions.holistic  # used to extract full body keypoints


    # frame_count = (frame_path.split('/')[-1]).split('.')[0]



    features_old = np.zeros((54, 2))
    
    # config = read_config('../config.yaml')
    
    feature_filters = Feature_Filters(
        face_filter=config['filters']['face_filter'],
        pose_filter=config['filters']['pose_filter'],
        lh_filter=config['filters']['lh_filter'],
        rh_filter=config['filters']['rh_filter']
    )
    feature_extractor = Feature_Extraction(filters=feature_filters)


    with mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2) as holistic:

        frame, results = feature_extractor.mediapipe_detection(frame, holistic)
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        status, face, pose, left_hand, right_hand = feature_extractor.extract_keypoints(results)
        if status == False:
            features = features_old
        else:
            features = feature_extractor.extract_features(face, pose, left_hand, right_hand, frame_width, frame_height)
            features_old = features


        print(features.shape)
        if np.sum(features[[26,29,32,35],:]) and np.sum(features[[42,45,48,51],:]) != 0.0:  # Checking blured frames                                
            print("saving features")

            npy_path = os.path.join(npy_save_dir, f"{frame_count}.npy")
            np.save(npy_path, features)
            
            return frame_count
        else:
            return None


def model_prediction(model,test_loader):
    predictions_dict = {}  # Dictionary to store inputs and their predictions
    total = 0
    # Ensure the model is in evaluation mode
    model.eval()
    model.to(device)

    # Iterate through test dataset
    for images in test_loader:
        # Extract the tensor from the list (since it's a list of inputs)
        images = images[0]  # Get the input tensor
        # print(images.shape)
        
        # Move images to the appropriate device (CPU/GPU)
        images = images.to(device)  # If using GPU, otherwise omit or use .to('cpu')

        # Forward pass to get logits/output
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = model(images)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Iterate through the batch and associate each input with its prediction
        for img, pred in zip(images, predicted):
            # Store input and prediction in the dictionary
            # If you want to store the raw image tensor, do so directly:
            predictions_dict[img.cpu()] = pred.cpu().item()  # Store input tensor and predicted label

            # If you prefer to store a simpler reference (e.g., index or image number), use:
            # predictions_dict[f"input_{total}"] = pred.cpu().item()  # Use a custom key instead of the raw image

        # Update total number of processed images
        total += images.size(0)

    # Now `predictions_dict` contains all the inputs and their corresponding predictions
    # print(len(predictions_dict))
    return predictions_dict




class Vector_models():
    def __init__(self,shift_orgin):
        
        
        self.config = read_config('config.yaml')

        self.shift_orgin = shift_orgin

        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load config
        # config = read_config('config.yaml')
        # input_dim = config['inference']["input_dim"]
        # hidden_dim = config['inference']["hidden_dim"]

        if shift_orgin:


            self.models = {}
            self.model_configs = {
                "right_fingertips_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/right_hand/right_fingertips_orientation.h5",
                "right_finger_closeness_to_face": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/right_hand/right_finger_closeness_to_face.h5",
                "right_fingers_joined": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/right_hand/right_fingers_joined.h5",
                "right_palm_position": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/right_hand/right_palm_position.h5",
                "right_elbow_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/right_hand/right_elbow_orientation.h5",
                "right_hand_position_along_body": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/right_hand/right_hand_position_along_body.h5",
                "right_forearm_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/right_hand/right_forearm_orientation.h5",
                "right_arm_folded": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/right_hand/right_arm_folded.h5",
                "hands_involved": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/two_hands/hands_involved.h5",
                "joined_hand_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/two_hands/joined_hand_orientation.h5",
                "relative_hand_height": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/two_hands/relative_hand_height.h5",
                "hand_synchronization": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/two_hands/hand_synchronization.h5",
                "left_arm_folded": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/left_hand/left_arm_folded.h5",
                "left_forearm_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/left_hand/left_forearm_orientation.h5",
                "left_elbow_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/left_hand/left_elbow_orientation.h5",
                "left_hand_position_along_body": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/left_hand/left_hand_position_along_body.h5",
                "left_fingertips_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/left_hand/left_fingertips_orientation.h5",
                "left_palm_position": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/left_hand/left_palm_position.h5",
                "left_fingers_joined": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/left_hand/left_fingers_joined.h5",
                "left_fingers_closeness_to_face": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/shifted_origin/left_hand/left_fingers_closeness_to_face.h5",
            }


        else:

            # Define model configurations
            self.models = {}
            self.model_configs = {
                "right_fingertips_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/right_fingertips_orientation.h5",
                "right_finger_closeness_to_face": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/right_finger_closeness_to_face.h5",
                "right_fingers_joined": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/right_fingers_joined.h5",
                "right_palm_position": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/right_palm_position.h5",
                "right_elbow_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/right_elbow_orientation.h5",
                "right_hand_position_along_body": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/right_hand_position_along_body.h5",
                "right_forearm_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/right_forearm_orientation.h5",
                "right_arm_folded": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/right_arm_folded.h5",
                "hands_involved": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/hands_involved.h5",
                "joined_hand_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/joined_hand_orientation.h5",
                "relative_hand_height": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/relative_hand_height.h5",
                "hand_synchronization": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/hand_synchronization.h5",
                "left_arm_folded": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/left_arm_folded.h5",
                "left_forearm_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/left_forearm_orientation.h5",
                "left_elbow_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/left_elbow_orientation.h5",
                "left_hand_position_along_body": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/left_hand_position_along_body.h5",
                "left_fingertips_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/left_fingertips_orientation.h5",
                "left_palm_position": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/left_palm_position.h5",
                "left_fingers_joined": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/left_fingers_joined.h5",
                "left_fingers_closeness_to_face": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/54keypoints/left_fingers_closeness_to_face.h5",
            }




        # Define models for each key in model_configs
        for model_name, path in self.model_configs.items():
            print(model_name)
            input_dim = self.config['input_dim_mapping'][model_name]
            output_dim = self.config['output_dim_mapping'][model_name]
            hidden_dim_key = self.config['hidden_dim_mapping'][model_name]

            # Initialize the model and load the state dictionary
            if model_name.split('_')[0] == 'right':
                # model = FeedforwardNeuralNetModel(input_dim, hidden_dim_key, output_dim).to(device)
                model = FeedforwardNeuralNetModelLeftHand(input_dim * 2, hidden_dim_key, output_dim).to(device)  # For Shifted Orgin
                model.load_state_dict(torch.load(path, map_location=device))
            else:
                model = FeedforwardNeuralNetModelLeftHand(input_dim * 2, hidden_dim_key, output_dim).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
            
            # Store model in the dictionary for later use
            self.models[model_name] = model
            print("----models loaded------")
    
    def model_predict(self, model_name, test_loader):
        # Ensure the model is in evaluation mode (no dropout, batchnorm etc.)
        model = self.models[model_name]
        model.to(device)
        model.eval()
        predictions_dict = model_prediction(model,test_loader)
        predictions_list = list(predictions_dict.values())
        return predictions_list



def read_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


#-------------------#
"""def dynamic_npy_read(init_vector_models,path,file_name,config):
    with open('/4TBHD/ISL/CodeBase/config.yaml', 'r') as file:
        config_keypoint = yaml.safe_load(file)
    features = np.load(path)

    selected_features = []
    for key in config_keypoint['feature_indices'][file_name]:
        # Construct the filter key from `filter_mapping`
        filter_key = f"{key}_filter"
        # Only add mapped values if the filter key exists in `filter_mapping`
        if filter_key in config_keypoint['filter_mapping']:
            for item in config_keypoint['feature_indices'][file_name][key]:
                # Append mapped value if it exists in the filter
                if item in config_keypoint['filter_mapping'][filter_key]:
                    selected_features.append(config_keypoint['filter_mapping'][filter_key][item])
    
    
    # z_tensor = torch.tensor(features[selected_features], dtype=torch.float32)
    z_tensor = (torch.tensor(features[selected_features], dtype=torch.float32)).reshape(1,-1)

    test_data = torch.utils.data.TensorDataset(z_tensor)
    test_loader = torch.utils.data.DataLoader(test_data, 1, shuffle=False)
    # left_hand_features_list = left_hand_features(test_loader,config['input_dim_mapping'][file_name],config['hidden_dim_mapping'][file_name],config['output_dim_mapping'][file_name],file_name)
    left_hand_features_list = init_vector_models.model_predict(file_name, test_loader)

    return left_hand_features_list
"""
#-----------------# 



def dynamic_npy_read_v1(init_vector_models,keypoints_arrays,file_name,classification_dict,config_keypoint): 
    batch_size = 100
    selected_feature_list=[]
    print("---->",file_name)

    # # For Shifted Orgin

    # for feature in keypoints_arrays:
    #     selected_features = []
    #     for key in config_keypoint['feature_indices'][file_name]:
    #         # Construct the filter key from `filter_mapping`
    #         filter_key = f"{key}_filter"
    #         # Only add mapped values if the filter key exists in `filter_mapping`
    #         if filter_key in config_keypoint['filter_mapping']:
    #             for item in config_keypoint['feature_indices'][file_name][key]:
    #                 # Append mapped value if it exists in the filter
    #                 if item in config_keypoint['filter_mapping'][filter_key]:
    #                     selected_features.append(config_keypoint['filter_mapping'][filter_key][item])
    #     selected_feature_list.append(feature[selected_features])
    # selected_feature_np = np.array(selected_feature_list)
    # selected_feature_flat = selected_feature_np.reshape(len(selected_feature_np),-1)
    # z_tensor = torch.tensor(selected_feature_flat, dtype=torch.float32)
    # test_data = torch.utils.data.TensorDataset(z_tensor)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False)
    
    # prediction_list = init_vector_models.model_predict(file_name, test_loader)
    # reconstructed_dict = {}
    # index = 0  # Used to track the position in `processed_list`

    # for key, value in classification_dict.items():
    #     sublist_length = len(value)  # Get the length of the list for this key
    #     reconstructed_dict[key] = prediction_list[index:index + sublist_length]
    #     index += sublist_length  # Move index forward by the length of the current list

    # return reconstructed_dict



    # Code for non-shifted Orgin

    # if "right" in file_name:
    for feature in keypoints_arrays:
        selected_features = []
        for key in config_keypoint['feature_indices'][file_name]:
            # Construct the filter key from `filter_mapping`
            filter_key = f"{key}_filter"
            # Only add mapped values if the filter key exists in `filter_mapping`
            if filter_key in config_keypoint['filter_mapping']:
                for item in config_keypoint['feature_indices'][file_name][key]:
                    # Append mapped value if it exists in the filter
                    if item in config_keypoint['filter_mapping'][filter_key]:
                        selected_features.append(config_keypoint['filter_mapping'][filter_key][item])
        selected_feature_list.append(feature[selected_features])
    selected_feature_np = np.array(selected_feature_list)
    selected_feature_flat = selected_feature_np.reshape(len(selected_feature_np),-1)
    z_tensor = torch.tensor(selected_feature_flat, dtype=torch.float32)
    test_data = torch.utils.data.TensorDataset(z_tensor)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False)
    # else:
    #     keypoints_arrays_np = np.array(keypoints_arrays)
    #     print(len(keypoints_arrays_np))

    #     z_tensor = torch.tensor(keypoints_arrays_np.reshape(len(keypoints_arrays_np),-1), dtype=torch.float32)
    #     test_data = torch.utils.data.TensorDataset(z_tensor)
    #     test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False)
    
    prediction_list = init_vector_models.model_predict(file_name, test_loader)
    reconstructed_dict = {}
    index = 0  # Used to track the position in `processed_list`

    for key, value in classification_dict.items():
        sublist_length = len(value)  # Get the length of the list for this key
        reconstructed_dict[key] = prediction_list[index:index + sublist_length]
        index += sublist_length  # Move index forward by the length of the current list

    return reconstructed_dict



def construct_seq_dictionary(config,classification_dict,dataset_size,shift_orgin,save_path = 'seq_ip_data.pkl'):
    start_time = time.time()
    init_vector_models = Vector_models(shift_orgin)
    feature_list,img_path=[],[]
    with open('/4TBHD/ISL/CodeBase/config.yaml', 'r') as file:
        config_keypoint = yaml.safe_load(file)
    #-------------------------------------------------------#
    # flat_path_list = [item for sublist in classification_dict.values() for item in sublist] # list of ndarray
    for sublist in classification_dict.keys():
        print((sublist))
        break
    
    if shift_orgin:
        keypoints_arrays = [shift_keypoints_to_new_orgin(path) for path in flat_path_list]

    else:
        # keypoints_arrays = [np.load(path) for path in flat_path_list] # Og
        keypoints_arrays = [path for path in flat_path_list]   # Dec 6 update

    print(keypoints_arrays[0])

    file_names = [
        "right_arm_folded",
        "right_forearm_orientation",
        "right_hand_position_along_body",
        "right_elbow_orientation",
        "right_palm_position",
        "right_fingers_joined",
        "right_fingertips_orientation",
        "right_finger_closeness_to_face",
        "left_arm_folded",  
        "left_forearm_orientation",
        "left_hand_position_along_body",    
        "left_elbow_orientation",  
        "left_palm_position",
        "left_fingers_joined",  
        "left_fingertips_orientation", 
        "left_fingers_closeness_to_face", 
        "hands_involved",
        "joined_hand_orientation",
        "relative_hand_height",
        "hand_synchronization"
    ]

    # Initialize the dictionary to store results
    results_dict = {}

    # Loop over file names and run the dynamic_npy_read_v1 function
    for file_name in file_names:
        # Assuming dynamic_npy_read_v1 is defined and available
        result = dynamic_npy_read_v1(init_vector_models, keypoints_arrays, file_name, classification_dict,config_keypoint)
        
        # Store the result in the dictionary with file_name as the key
        results_dict[file_name] = result

    # The results_dict now contains the results for all file names as keys
    # print(len(results_dict))
    #------------------------------------------#
    # res_dict = dynamic_npy_read_v1(init_vector_models,keypoints_arrays,"left_arm_folded",classification_dict)
    # print("-----",len(res_dict))
    # Extract video names (assuming all keys have the same video names)
    vid_name = list(next(iter(results_dict.values())).keys())

    # Initialize result vector
    result_vector = []

    # Iterate over video names
    for vid in vid_name:
        combined_vector = []
        # Iterate over each key in overall_dict
        for key in results_dict.keys():
            # Get the list of values for the current key and video
            values = results_dict[key][vid]
            combined_vector.append(values)

        # Transpose the combined lists to pair elements across keys
        paired_vector = list(zip(*combined_vector))
        result_vector.append(paired_vector)
    feature_list = [list(tup) for tup in result_vector]
    # Output the results
    print("Result Video Names:", vid_name)
    print("Result Vector:", type(feature_list[0]))

    index = 0  # Used to track the position in `processed_list`
    for key, value in classification_dict.items():
        sublist_length = len(value)  # Get the length of the list for this key
        img_path.append(flat_path_list[index:index + sublist_length])
        index += sublist_length  # Move index forward by the length of the current list

    labels = [] 
    vid_name = []
    for vid in classification_dict.keys():
        labels.append(vid.split("_")[-2])   # Later Kindly change to 1  ##### Remember ########################
        vid_name.append(vid)
    # Example DataFrame
    seq_ip_data = {'vid_name': vid_name,
            'feature_list': feature_list,
            'labels': labels,
            'img_path':img_path}
    
    if dataset_size == 'reduced':

        original_list = seq_ip_data['feature_list']
        vid_name = seq_ip_data['vid_name']
        labels = seq_ip_data['labels']
        original_img_path = seq_ip_data['img_path']
        new_vid_list,new_img_path = [],[]
        
        # Loop through the original list
        for i in range(len(original_list)):
            new_img_list = []
            new_path_list = []
            for j in range(len(original_list[i])):
                # Append the element to new_list only if it is not the same as the previous one
                if j == 0 or original_list[i][j] != original_list[i][j - 1]:
                    new_img_list.append(original_list[i][j])
                    new_path_list.append(original_img_path[i][j])
            new_vid_list.append(new_img_list)
            new_img_path.append(new_path_list)

        seq_ip_data = {'vid_name': vid_name,
                'feature_list': new_vid_list,
                'labels': labels,
                'img_path':new_img_path}

        with open(save_path, 'wb') as f:
            pickle.dump(seq_ip_data, f)
    else:

        with open(save_path, 'wb') as f:
            pickle.dump(seq_ip_data, f)
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time} seconds")



def process_video(video_path, config, npy_base_dir, npy_dict, lock):
    """Process a single video: extract frames, generate npy files, and update the dictionary."""
    video_name = os.path.basename(video_path).split('.')[0].lower()
    # unique_tmp_dir = os.path.join(frame_tmp_base_dir, video_name)
    # os.makedirs(unique_tmp_dir, exist_ok=True)

        
    npy_save_dir = os.path.join(npy_base_dir, video_name)
    os.makedirs(npy_save_dir,exist_ok=True)

    try:
        # keyframes = extract_start_end_frames_with_decrementing_threshold_function(video_path)
        extract_seq_of_frames(config,video_path, npy_save_dir,lock,video_name,npy_dict)
        
        
    finally:
        print("Hola done")  # Clean up the temporary directory after use

def construct_npy_dictionary(config, npy_base_dir, test_data_path, npy_pickle_path='test.pkl'):
    """Construct npy dictionary with video_name as key and npy file paths as values."""
    npy_dict = defaultdict(list)
    lock = Lock()

    video_paths = [
        os.path.join(test_data_path, Signers, folders, video)
        for Signers in os.listdir(test_data_path)
        for folders in os.listdir(os.path.join(test_data_path,Signers))  # Replace with actual folder logic if needed
        for video in os.listdir(os.path.join(test_data_path, Signers, folders))
    ]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_video, video_path, config, npy_base_dir, npy_dict, lock)
            for video_path in video_paths
        ]
        for future in as_completed(futures):
            future.result()  # Wait for all threads to complete
    
    with open(npy_pickle_path, 'wb') as f:
        pickle.dump(dict(npy_dict), f)



def balance_classes(df, target_column):
    # Determine the maximum count of samples in any class
    max_count = 10

    # Resample each class to the max_count
    balanced_dfs = []
    for label, group in df.groupby(target_column):
        if len(group) < max_count:
            # Upsample
            balanced_group = resample(group, replace=True, n_samples=max_count, random_state=42)
        else:
            # Downsample
            balanced_group = group.sample(n=max_count, random_state=42)
        balanced_dfs.append(balanced_group)

    # Combine balanced dataframes
    balanced_df = pd.concat(balanced_dfs).reset_index(drop=True)
    return balanced_df