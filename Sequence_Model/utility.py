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



def extract_npy_files(config,frame_path,npy_save_dir):

    mp_holistic = mp.solutions.holistic  # used to extract full body keypoints


    frame_count = (frame_path.split('/')[-1]).split('.')[0]



    features_old = np.zeros((54, 2))
    
    # config = read_config('../config.yaml')
    
    feature_filters = Feature_Filters(
        face_filter=config['filters']['face_filter'],
        pose_filter=config['filters']['pose_filter'],
        lh_filter=config['filters']['lh_filter'],
        rh_filter=config['filters']['rh_filter']
    )
    feature_extractor = Feature_Extraction(filters=feature_filters)


    frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
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

        # if np.sum(features[-4]) != 0.0:
        #     npy_path = os.path.join(npy_save_dir, f"{frame_count}.npy")
        #     np.save(npy_path, features)

        # if np.sum(features[-4]) and np.sum(features[32:35]) != 0.0:  # Checking blured frames                

        print(features.shape)
        if np.sum(features[[26,29,32,35],:]) and np.sum(features[[42,45,48,51],:]) != 0.0:  # Checking blured frames                                


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
                "right_fingertips_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/multi_people_models_v3/right_fingertips_orientation.h5",
                "right_finger_closeness_to_face": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/multi_people_models_v3/right_finger_closeness_to_face.h5",
                "right_fingers_joined": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/multi_people_models_v3/right_fingers_joined.h5",
                "right_palm_position": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/multi_people_models_v3/right_palm_position.h5",
                "right_elbow_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/multi_people_models_v3/right_elbow_orientation.h5",
                "right_hand_position_along_body": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/multi_people_models_v3/right_hand_position_along_body.h5",
                "right_forearm_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/multi_people_models_v3/right_forearm_orientation.h5",
                "right_arm_folded": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/multi_people_models_v3/right_arm_folded.h5",
                "hands_involved": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/two_hands_v1/hands_involved.h5",
                "joined_hand_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/two_hands_v1/joined_hand_orientation.h5",
                "relative_hand_height": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/two_hands_v1/relative_hand_height.h5",
                "hand_synchronization": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/two_hands_v1/hand_synchronization.h5",
                "left_arm_folded": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/left_v1/left_arm_folded.h5",
                "left_forearm_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/left_v1/left_forearm_orientation.h5",
                "left_elbow_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/left_v1/left_elbow_orientation.h5",
                "left_hand_position_along_body": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/left_v1/left_hand_position_along_body.h5",
                "left_fingertips_orientation": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/left_v1/left_fingertips_orientation.h5",
                "left_palm_position": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/left_v1/left_palm_position.h5",
                "left_fingers_joined": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/left_v1/left_fingers_joined.h5",
                "left_fingers_closeness_to_face": "/4TBHD/ISL/CodeBase/Model_Dir/vector_models/left_v1/left_fingers_closeness_to_face.h5",
            }




        # Define models for each key in model_configs
        for model_name, path in self.model_configs.items():
            input_dim = self.config['input_dim_mapping'][model_name]
            output_dim = self.config['output_dim_mapping'][model_name]
            hidden_dim_key = self.config['hidden_dim_mapping'][model_name]

            # Initialize the model and load the state dictionary
            if model_name.split('_')[0] == 'right':
                model = FeedforwardNeuralNetModel(input_dim, hidden_dim_key, output_dim).to(device)
                # model = FeedforwardNeuralNetModelLeftHand(input_dim * 2, hidden_dim_key, output_dim).to(device)  # For Shifted Orgin
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

"""def left_hand_features(test_loader , input_dim, hidden_dim, output_dim,file_name):
    # print(file_name)
    # model = FeedforwardNeuralNetModelLeftHand(input_dim*2, hidden_dim, output_dim,dropout_prob = 0.5)
    model.to(device)
    
    # criterion = nn.CrossEntropyLoss()
    # model_{file_name}.load_state_dict(torch.load(f'/4TBHD/ISL/CodeBase/Model_Dir/vector_models/left_v1/{file_name}.h5'))
    model = getattr(self, file_name)
    predictions_dict = model_prediction(model,test_loader)
    predictions_list = list(predictions_dict.values())
    return predictions_list

def two_hands_features(test_loader,model,file_name ):# input_dim, hidden_dim, output_dim,file_name):
    # model = FeedforwardNeuralNetModelLeftHand(input_dim, hidden_dim, output_dim,dropout_prob = 0.5)
    model.to(device)
    # criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(f'/4TBHD/ISL/CodeBase/Model_Dir/vector_models/two_hands_v1/{file_name}.h5'))
    predictions_dict = model_prediction(model,test_loader)
    predictions_list = list(predictions_dict.values())
    return predictions_list
"""


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

    if "left" in file_name:
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
    else:
        keypoints_arrays_np = np.array(keypoints_arrays)
        z_tensor = torch.tensor(keypoints_arrays_np.reshape(len(keypoints_arrays_np),-1), dtype=torch.float32)
        test_data = torch.utils.data.TensorDataset(z_tensor)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False)
    
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






























"""
def construct_seq_dictionary(config,classification_dict,dataset_size,save_path = 'seq_ip_data.pkl'):

    # models = [model_right_arm_folded, model_right_forearm_orientation, model_right_hand_position_along_body,model_right_elbow_orientation,model_right_palm_position,model_right_fingers_joined,
    #           model_rigconstructht_finger_orientation,model_right_finger_closeness_to_face,model_hands_involved,model_joined_hand_orientation,model_relative_hand_height,model_hand_synchronization]
construct
    # # Initialize the MultiModelInference class
    # multi_inference = MultiModelInference(models)

    # # # Load all models to the GPU
    # multi_inference.load_all_models()
construct
    # loop here for every image
    init_vector_models = Vector_models()
    feature_list,img_path=[],[]
    for vid in tqdm(classification_dict.keys()):
        vector_list = []
        path_list = []
        for img_idx in range(len(classification_dict[vid])):
        # print("length of classification_dict",len(classification_dict))
            path = classification_dict[vid][img_idx].replace('onlyface','keypoints').replace('jpg','npy')
            

            keypoint = np.load(path).reshape(1,-1)
            #-----------------# 
            left_arm_folded_list = dynamic_npy_read(init_vector_models,path,'left_arm_folded',config)
            left_forearm_orientation_list = dynamic_npy_read(init_vector_models,path,'left_forearm_orientation',config)
            left_elbow_orientation_list = dynamic_npy_read(init_vector_models,path,'left_elbow_orientation',config)
            left_hand_position_along_body_list = dynamic_npy_read(init_vector_models,path,'left_hand_position_along_body',config)
            left_fingertips_orientation_list = dynamic_npy_read(init_vector_models,path,'left_fingertips_orientation',config)
            left_palm_position_list = dynamic_npy_read(init_vector_models,path,'left_palm_position',config)
            left_fingers_joined_list = dynamic_npy_read(init_vector_models,path,'left_fingers_joined',config)
            left_fingers_closeness_to_face_list = dynamic_npy_read(init_vector_models,path,'left_fingers_closeness_to_face',config)

            # left_arm_folded_tensor = torch.tensor(left_arm_folded_keypoint, dtype=torch.float32)
            # left_forearm_orientation_tensor = torch.tensor(left_forearm_orientation_keypoint, dtype=torch.float32)
            # left_elbow_orientation_tensor = torch.tensor(left_elbow_orientation_keypoint, dtype=torch.float32)
            # left_hand_position_along_body_tensor = torch.tensor(left_hand_position_along_body_keypoint, dtype=torch.float32)
            # left_fingertips_orientation_tensor = torch.tensor(left_fingertips_orientation_keypoint, dtype=torch.float32)
            # left_palm_position_tensor = torch.tensor(left_palm_position_keypoint, dtype=torch.float32)
            # left_fingers_joined_tensor = torch.tensor(left_fingers_joined_keypoint, dtype=torch.float32)
            # left_fingers_closeness_to_face_tensor = torch.tensor(left_fingers_closeness_to_face_keypoint, dtype=torch.float32)


            #-----------------#

            z_tensor = torch.tensor(keypoint, dtype=torch.float32)
            test_data = torch.utils.data.TensorDataset(z_tensor)
            test_loader = torch.utils.data.DataLoader(test_data, 1, shuffle=False)
            
            right_arm_folded_list = init_vector_models.model_predict("right_arm_folded", test_loader)
            right_forearm_orientation_list = init_vector_models.model_predict("right_forearm_orientation", test_loader)
            right_elbow_orientation_list = init_vector_models.model_predict("right_elbow_orientation", test_loader)
            right_hand_position_along_body_list = init_vector_models.model_predict("right_hand_position_along_body", test_loader)
            right_fingertips_orientation_list = init_vector_models.model_predict("right_fingertips_orientation", test_loader)
            right_palm_position_list = init_vector_models.model_predict("right_palm_position", test_loader)
            right_fingers_joined_list = init_vector_models.model_predict("right_fingers_joined", test_loader)
            right_finger_closeness_to_face_list = init_vector_models.model_predict("right_finger_closeness_to_face", test_loader)
            hands_involved_list = init_vector_models.model_predict("hands_involved", test_loader)
            joined_hand_orientation_list = init_vector_models.model_predict("joined_hand_orientation", test_loader)
            relative_hand_height_list = init_vector_models.model_predict("relative_hand_height", test_loader)
            hand_synchronization_list = init_vector_models.model_predict("hand_synchronization", test_loader)

            # right_arm_folded_list = right_arm_folded(test_loader,model_right_arm_folded) #input_dim,hidden_dim,config['output_dim_mapping']["right_arm_folded"])
            # right_forearm_orientation_list = right_forearm_orientation(test_loader,model_right_forearm_orientation) #input_dim,hidden_dim,config['output_dim_mapping']["right_forearm_orientation"])
            # right_hand_position_along_body_list = right_hand_position_along_body(test_loader,model_right_hand_position_along_body)#input_dim,hidden_dim,config['output_dim_mapping']["right_hand_position_along_body"])
            # right_elbow_orientation_list = right_elbow_orientation(test_loader,model_right_elbow_orientation)#input_dim,hidden_dim,config['output_dim_mapping']["right_elbow_orientation"])
            # right_palm_position_list = right_palm_position(test_loader,model_right_palm_position)#input_dim,hidden_dim,config['output_dim_mapping']["right_palm_position"])
            # right_fingers_joined_list = right_fingers_joined(test_loader,model_right_fingers_joined)#input_dim,hidden_dim,config['output_dim_mapping']["right_fingers_joined"])
            # right_finger_orientation_list = right_finger_orientation(test_loader,model_right_finger_orientation)#input_dim,hidden_dim,config['output_dim_mapping']["right_finger_orientation"])
            # right_finger_closeness_to_face_list = right_finger_closeness_to_face(test_loader,model_right_finger_closeness_to_face)#input_dim,hidden_dim,config['output_dim_mapping']["right_finger_closeness_to_face"])
            
            # hands_involved_list = two_hands_features(test_loader,model_hands_involved, "hands_involved") #input_dim,config['hidden_dim_mapping']['hands_involved'],config['output_dim_mapping']["hands_involved"],"hands_involved")
            # joined_hand_orientation_list = two_hands_features(test_loader,model_joined_hand_orientation,"joined_hand_orientation")#"input_dim,config['hidden_dim_mapping']['joined_hand_orientation'],config['output_dim_mapping']["joined_hand_orientation"],"joined_hand_orientation")
            # relative_hand_height_list = two_hands_features(test_loader,model_relative_hand_height,'relative_hand_height') #input_dim,config['hidden_dim_mapping']['relative_hand_height'],config['output_dim_mapping']["relative_hand_height"],"relative_hand_height")
            # hand_synchronization_list = two_hands_features(test_loader, model_hand_synchronization,'hand_synchronization')#input_dim,config['hidden_dim_mapping']['hand_synchronization'],config['output_dim_mapping']["hand_synchronization"],"hand_synchronization")

            feature_vector =  right_arm_folded_list + right_forearm_orientation_list + right_hand_position_along_body_list + right_elbow_orientation_list + right_palm_position_list + right_fingers_joined_list + right_fingertips_orientation_list + right_finger_closeness_to_face_list + left_arm_folded_list + left_forearm_orientation_list + left_elbow_orientation_list + left_hand_position_along_body_list + left_fingertips_orientation_list + left_palm_position_list + left_fingers_joined_list + left_fingers_closeness_to_face_list + hands_involved_list + joined_hand_orientation_list + relative_hand_height_list + hand_synchronization_list
            # print("feature_vector",feature_vector)
            vector_list.append(feature_vector)
            path_list.append(classification_dict[vid][img_idx])
        # print("vector_list",len(vector_list))
        feature_list.append(vector_list)
        img_path.append(path_list)
    labels = [] 
    vid_name = []
    for vid in classification_dict.keys():
        labels.append(vid.split("_")[1])
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
            new_vid_list.constructappend(new_img_list)
            new_img_path.append(new_path_list)
        
        # Output the new list without consecutive duplicates
        # print(len(new_vid_list))
        
        seq_ip_data = {'vid_name': vid_name,
                'feature_list': new_vid_list,
                'labels': labels,
                'img_path':new_img_path}

        with open(save_path, 'wb') as f:
            pickle.dump(seq_ip_data, f)
    else:

        with open(save_path, 'wb') as f:
            pickle.dump(seq_ip_data, f)

    # multi_inference.clear_gpu_memory()

"""
def construct_npy_dictionary(config,npy_base_dir,frame_tmp_dir,test_data_path,npy_pickle_path = 'npy_dict_all.pkl'):

    """"
    This code is to construct npy dictionary of with key as video_name and value as list of npy paths.
    
    """

    npy_dict = {}
    for Signers in os.listdir(test_data_path):
        # for folders in os.listdir(os.path.join(test_data_path,Signers)):                 # Commented for ai4bharat Testing TODO : Change to GPT code to dynamically handle any directory structure
            # if folders in videos_to_extract:
            folders = ''
            for video in os.listdir(os.path.join(test_data_path,Signers,folders)):
                video_path = os.path.join(test_data_path,Signers,folders,video)
    #             # print(video_path)
                keyframes = extract_start_end_frames_with_decrementing_threshold_function(video_path) # For one video
    #             # print(keyframes)
                clear_tmp_directory(directory_path=frame_tmp_dir) 
                extract_seq_of_frames(video_path ,keyframes,frame_tmp_dir)
            
                for img in sorted(os.listdir(frame_tmp_dir),key=lambda x: int(x.split('.')[0])):
                    video_name = video
                    img_path = os.path.join(frame_tmp_dir,img)
                    npy_save_dir = os.path.join(npy_base_dir,(video_name.split('.')[0]).lower())
                    os.makedirs(npy_save_dir,exist_ok=True)
                    frame_no = extract_npy_files(config,img_path,npy_save_dir)
                    if frame_no != None:
                        key = video_name.split('.')[0].lower()
                        if key not in npy_dict:
                            npy_dict[key] = []
                        # frame_no = (img_path.split('/')[-1]).split('.')[0]
                        # print("Another loop",(img_path.split('/')[-1]).split('.')[0])                    
                        npy_dict[key].append(f'{npy_save_dir}/{frame_no}.npy')
                    # print("dict values",npy_dict)
                    else:
                        continue

    #             break

    with open(npy_pickle_path, 'wb') as f:
        pickle.dump(npy_dict, f)


def construct_npy_dictionary_one_video_multithread(config,video_path):


    # Extract keyframes for one video

    keyframes = extract_start_end_frames_with_decrementing_threshold_function(video_path)
    
    pass 





def construct_npy_file(config,npy_base_dir,frame_tmp_dir,video_path,npy_pickle_path = 'npy_dict_all.pkl'):

    npy_dict = {}

    keyframes = extract_start_end_frames(video_path)
    clear_tmp_directory(directory_path=frame_tmp_dir)
    extract_seq_of_frames(video_path ,keyframes,frame_tmp_dir)

    for img in sorted(os.listdir(frame_tmp_dir),key=lambda x: int(x.split('.')[0])):
        video_name = video_path.split('/')[-1].split('.')[0].lower()
        img_path = os.path.join(frame_tmp_dir,img)
        npy_save_dir = os.path.join(npy_base_dir,(video_name.split('.')[0]).lower())
        os.makedirs(npy_save_dir,exist_ok=True)
        extract_npy_files(config,img_path,npy_save_dir)
        key = video_name.split('.')[0].lower()
        if key not in npy_dict:
            npy_dict[key] = []
        frame_no = (img_path.split('/')[-1]).split('.')[0]
        # print("Another loop",(img_path.split('/')[-1]).split('.')[0])                    
        npy_dict[key].append(f'{npy_save_dir}/{frame_no}.npy')
        # print("dict values",npy_dict)

    #             break

    with open(npy_pickle_path, 'wb') as f:
        pickle.dump(npy_dict, f)

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