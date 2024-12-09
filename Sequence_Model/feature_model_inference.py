import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import os
import re
import yaml
import pandas as pd
import csv
from model import *
import pickle


def get_X_and_Y_value(csv_file_path):
 
    X_label,X_file_name  = [],[]
 
    data = pd.read_csv(csv_file_path)
 
    print(data.columns)
    for index, row in data.iterrows():
        image_name = row[data.columns[0]]
        X_file_name.append(image_name)
        X_label.append(np.load(image_name))

    return X_label,X_file_name



def read_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def model_prediction(model,test_loader):
    predictions_dict = {}  
    total = 0
    model.eval()

    for images in test_loader:
        images = images[0]  # Get the input tensor
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        for img, pred in zip(images, predicted):
            predictions_dict[img.cpu()] = pred.cpu().item()  # Store input tensor and predicted label
        total += images.size(0)
    print(len(predictions_dict))
    return predictions_dict

 
def model_inference(model_path,data, input_dim, hidden_dim, output_dim):
    model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(f'/4TBHD/ISL/CodeBase/vector_models/{model_path}'))
    predictions_dict = model_prediction(model,data)
    predictions_list = list(predictions_dict.values())
    return predictions_list

def model_inference_sandy(model_name):

    model_name_split = model_name.split('/')[-1]
    output_dim = config["output_dim_mapping"][model_name_split]
    model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    # criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(f'/4TBHD/ISL/CodeBase/vector_models/{model_name}.h5'))

    data_z_flat = data.reshape(1,-1)


    print("data flat",data_z_flat.shape)

    z_tensor = torch.tensor(data_z_flat, dtype=torch.float32)
    print(z_tensor.shape)
    model.eval()

    with torch.no_grad(): 
        outputs = model(z_tensor)

    _, predicted = torch.max(outputs.data, 1)

    print(predicted)

    prediction_dictionary_mapping = config['prediction_label_dict'][model_name]

    print(prediction_dictionary_mapping)

    predicted_str = prediction_dictionary_mapping[(predicted.item())]

    print(predicted.item())
    print(predicted_str)


if __name__ == '__main__':

    config = read_config('config.yaml')

 


    input_dim =  config['inference']["input_dim"]
    hidden_dim =  config['inference']["hidden_dim"]



    model_list = ["10kmodels/right_arm_folded",
                    "10kmodels/right_forearm_orientation",
                    "10kmodels/right_hand_position_along_body",
                    "10kmodels/right_elbow_orientation",
                    "10kmodels/right_palm_position",
                    "10kmodels/right_fingers_joined",
                    "multi_people_models_1/right_fingertips_orientation",
                    "10kmodels/right_finger_closeness_to_face"
                    ]
  
        
    pickle_file_save_dir = '/4TBHD/ISL/CodeBase'

    file_path = f'{pickle_file_save_dir}/seq_dataset.pkl'
    if os.path.getsize(file_path) > 0:
        with open(file_path, 'rb') as f:
            classification_dict = pickle.load(f)
    else:
        print("The file is empty.")
    
    print(classification_dict["stephen_hello_6"][2])





    path = classification_dict["stephen_hello_6"][2]
    path = path.replace('onlyface','keypoints').replace('jpg','npy')

    data = np.load(path)

    print(data.shape)

    model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load(f'/4TBHD/ISL/CodeBase/vector_models/10kmodels/{model_name}.h5'))

    data_z_flat = data.reshape(1,-1)


    print("data flat",data_z_flat.shape)

    z_tensor = torch.tensor(data_z_flat, dtype=torch.float32)
    print(z_tensor.shape)
    model.eval()

    with torch.no_grad(): 
        outputs = model(z_tensor)

    _, predicted = torch.max(outputs.data, 1)

    print(predicted)

    prediction_dictionary_mapping = config['prediction_label_dict'][model_name]

    print(prediction_dictionary_mapping)

    predicted_str = prediction_dictionary_mapping[(predicted.item())]

    print(predicted.item())
    print(predicted_str)


    """
    
    1x8 - vector embedding

    1. arm folded
    2. forearm orientation
    3. hand_position along body
    4. elbow orientation
    5. right palm position
    6. right fingers joined
    7. fingertips orientation
    8. finger closeness to face
    
    """




    # for img, pred in zip(images, predicted):

    #     predictions_dict[img.cpu()] = pred.cpu().item()

    # total += images.size(0)

    # print(len(predictions_dict))    # data_z_flat = data.reshape(len(data), -1)


    # # List of all inputs (keys in the dictionary)
    # prediction_label_dict = {
        
    #     # right arm folded
    #     # 0:'partially folded',
    #     # 1: 'folded',
    #     # 2: 'rest',

    #     # right forearm orientation
    #     # 0:  "horizontal",
    #     # 1: "vertical",
    #     # 2: "lower diagonal",
    #     # 3: "upper diagonal",

    #     #  finger closeness to face
    #     # 0: "eyes",
    #     # 1: "ears",
    #     # 2: "nose",
    #     # 3: "lips",
    #     # 4: "chin",
    #     # 5: "cheeks",
    #     # 6: "none",
    #     # 7: "hair",
    #     # 8: "forehead"

    
    #     # # fingers joined
    #     # 0: 'joined',
    #     # 1: 'notjoined',
 
    #     #finger orientation
    #     # 0: 'open',
    #     # 1: 'closed',
    #     # 2: 'thumb alone open',
    #     # 3: 'index finger open',
    #     # 4: 'partially joined'
 
    #     #position along body
    #     # 0: 'chest',
    #     # 1: 'face',
    #     # 2: 'abdomen',
    #     # 3: 'belowabdomen'

    #     # right palm position
    #     # 0: "towards body",
    #     # 1: "upwards",
    #     # 2: "away from body",
    #     # 3: "downwards",
    #     # 4: "towards left",
    #     # 5: "towards right"
    # }

    # inputs_list = [s.replace('keypoints', 'onlyface').replace('.npy','.jpg') for s in X_file_name]
    # predictions_list = list(predictions_dict.values())
    # string_predictions_list = [prediction_label_dict[prediction] for prediction in predictions_list]


    # # Convert inputs_list and predictions_list to a pandas DataFrame
    # data = {
    #     "Input": [input_item.cpu().tolist() if isinstance(input_item, torch.Tensor) else str(input_item) for input_item in inputs_list],
    #     "Prediction": string_predictions_list
    # }

    # df = pd.DataFrame(data)

    # os.makedirs('Test_Dataset_Prediction',exist_ok=True)
    # # Path to save the Excel file
    # excel_file_path = config['inference']['excel_file_path']
    
    # # Save the DataFrame to an Excel file
    # df.to_csv(excel_file_path, index=False)  # index=False to avoid writing row numbers

    # # # Now, the Excel file contains the inputs and predictions
    # print(f"Excel file saved at: {excel_file_path}")


