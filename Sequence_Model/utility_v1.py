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
from load_model_gpu import *
from concurrent.futures import ThreadPoolExecutor, as_completed

if torch.cuda.is_available():  
    device = "cuda" 
else:  
    device = "cpu"  

# device = torch.device('cpu')

torch.set_default_tensor_type(torch.cuda.FloatTensor)

def calculate_model_classwise_accuracy(model, loss_fn, loader, device, num_classes):
    model.to(device)
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

class Vector_models():
    def __init__(self):
        self.config = read_config('config.yaml')

        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

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
            print("model_name",model_name)
            input_dim = self.config['input_dim_mapping'][model_name]
            output_dim = self.config['output_dim_mapping'][model_name]
            hidden_dim_key = self.config['hidden_dim_mapping'][model_name]

            # Initialize the model and load the state dictionary
            if model_name.split('_')[0] == 'right':
                model = FeedforwardNeuralNetModel(input_dim, hidden_dim_key, output_dim).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
            else:
                model = FeedforwardNeuralNetModelLeftHand(input_dim * 2, hidden_dim_key, output_dim).to(device)
                model.load_state_dict(torch.load(path, map_location=device))

            self.models[model_name] = model.eval()
            # print("----models loaded------")
    
    def model_predict(self, model_name, test_loader):
        # Ensure the model is in evaluation mode (no dropout, batchnorm etc.)
        model = self.models[model_name]
        predictions_list = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs[0].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions_list.extend(predicted.cpu().tolist())
        return predictions_list

def dynamic_npy_read_v1(init_vector_models,keypoints_arrays,file_name,classification_dict):
    with open('/4TBHD/ISL/CodeBase/config.yaml', 'r') as file:
        config_keypoint = yaml.safe_load(file)
    batch_size = 100
    selected_feature_list=[]
    print("---->",file_name)
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


from concurrent.futures import ThreadPoolExecutor
import time
def construct_seq_dictionary(config,classification_dict,dataset_size,save_path = 'seq_ip_data.pkl'):
    start_time = time.time()
    init_vector_models = Vector_models()
    feature_list,img_path=[],[]
    
    #-------------------------------------------------------#
    flat_path_list = [item for sublist in classification_dict.values() for item in sublist]
    keypoints_arrays = [np.load(path) for path in flat_path_list]
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
        result = dynamic_npy_read_v1(init_vector_models, keypoints_arrays, file_name, classification_dict)
        
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

def construct_npy_dictionary(config,npy_base_dir,frame_tmp_dir,test_data_path,npy_pickle_path = 'npy_dict_all.pkl'):

    npy_dict = {}
    for Signers in os.listdir(test_data_path):
        for folders in os.listdir(os.path.join(test_data_path,Signers)):
            # if folders in videos_to_extract:
            for video in os.listdir(os.path.join(test_data_path,Signers,folders)):
                video_path = os.path.join(test_data_path,Signers,folders,video)
    #             # print(video_path)
                keyframes = extract_start_end_frames_with_decrementing_threshold_function(video_path)
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
                        npy_dict[key].append(f'{npy_save_dir}/{frame_no}.npy')
                    else:
                        continue

    with open(npy_pickle_path, 'wb') as f:
        pickle.dump(npy_dict, f)


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