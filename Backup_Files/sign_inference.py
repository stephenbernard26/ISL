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
from collections import Counter

from sequence_classifier import *
from custom_dataloader import *

def read_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

if torch.cuda.is_available():  
    device = "cuda" 
else:  
    device = "cpu"  

print("device",device)

def model_prediction(model,test_loader):
    predictions_dict = {}  # Dictionary to store inputs and their predictions
    total = 0
    # Ensure the model is in evaluation mode
    model.eval()

    # Iterate through test dataset
    for images in test_loader:
        # Extract the tensor from the list (since it's a list of inputs)
        images = images[0]  # Get the input tensor
        
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

def right_arm_folded(test_loader,input_dim, hidden_dim, output_dim):
    model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load('/4TBHD/ISL/CodeBase/vector_models/10kmodels_v2/right_arm_folded.h5'))
    predictions_dict = model_prediction(model,test_loader)
    # prediction_label_dict = {
    # 0:'partially folded',
    # 1: 'folded',
    # }
    predictions_list = list(predictions_dict.values())
    # string_predictions_list = [prediction_label_dict[prediction] for prediction in predictions_list]
    return predictions_list
    
def right_forearm_orientation(test_loader, input_dim, hidden_dim, output_dim):
    model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load('/4TBHD/ISL/CodeBase/vector_models/10kmodels_v2/right_forearm_orientation_1.h5'))
    predictions_dict = model_prediction(model,test_loader)
    predictions_list = list(predictions_dict.values())
    return predictions_list

def right_hand_position_along_body(test_loader, input_dim, hidden_dim, output_dim):
    model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load('/4TBHD/ISL/CodeBase/vector_models/10kmodels_v2/right_hand_position_along_body.h5'))
    predictions_dict = model_prediction(model,test_loader)
    predictions_list = list(predictions_dict.values())
    return predictions_list

def right_elbow_orientation(test_loader, input_dim, hidden_dim, output_dim):
    model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load('/4TBHD/ISL/CodeBase/vector_models/10kmodels_v2/right_elbow_orientation.h5'))
    predictions_dict = model_prediction(model,test_loader)
    predictions_list = list(predictions_dict.values())
    return predictions_list

def right_palm_position(test_loader, input_dim, hidden_dim, output_dim):
    model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load('/4TBHD/ISL/CodeBase/vector_models/10kmodels_v2/right_palm_position.h5'))
    predictions_dict = model_prediction(model,test_loader)
    predictions_list = list(predictions_dict.values())
    return predictions_list

def right_fingers_joined(test_loader, input_dim, hidden_dim, output_dim):
    model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load('/4TBHD/ISL/CodeBase/vector_models/10kmodels_v2/right_fingers_joined.h5'))
    predictions_dict = model_prediction(model,test_loader)
    predictions_list = list(predictions_dict.values())
    return predictions_list

def right_finger_orientation(test_loader, input_dim, hidden_dim, output_dim):
    model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load('/4TBHD/ISL/CodeBase/vector_models/10kmodels_v2/right_fingertips_orientation.h5'))
    predictions_dict = model_prediction(model,test_loader)
    predictions_list = list(predictions_dict.values())
    return predictions_list

def right_finger_closeness_to_face(test_loader, input_dim, hidden_dim, output_dim):
    model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load('/4TBHD/ISL/CodeBase/vector_models/10kmodels_v2/right_finger_closeness_to_face.h5'))
    predictions_dict = model_prediction(model,test_loader)
    predictions_list = list(predictions_dict.values())
    return predictions_list

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

if __name__ == '__main__':
  
  config = read_config('config.yaml')
  input_dim =  config['inference']["input_dim"]
  hidden_dim =  config['inference']["hidden_dim"]


  
  pickle_file_save_dir = '/4TBHD/ISL/CodeBase'

  file_path = f'npy_dict_all.pkl'
  if os.path.getsize(file_path) > 0:
      with open(file_path, 'rb') as f:
          classification_dict = pickle.load(f)
  else:
      print("The file is empty.")
  
  # loop here for every image
  feature_list=[]
  for vid in classification_dict.keys():
    vector_list = []
    for img in range(len(classification_dict[vid])):
      # print("length of classification_dict",len(classification_dict))
      path = classification_dict[vid][img].replace('onlyface','keypoints').replace('jpg','npy')
      print("path",path)
      keypoint = np.load(path).reshape(1,-1)
      z_tensor = torch.tensor(keypoint, dtype=torch.float32)

      test_data = torch.utils.data.TensorDataset(z_tensor)
      test_loader = torch.utils.data.DataLoader(test_data, 1, shuffle=False)
      
      right_arm_folded_list = right_arm_folded(test_loader,input_dim,hidden_dim,config['output_dim_mapping']["right_arm_folded"])
      right_forearm_orientation_list = right_forearm_orientation(test_loader,input_dim,hidden_dim,config['output_dim_mapping']["right_forearm_orientation"])
      right_hand_position_along_body_list = right_hand_position_along_body(test_loader,input_dim,hidden_dim,config['output_dim_mapping']["right_hand_position_along_body"])
      right_elbow_orientation_list = right_elbow_orientation(test_loader,input_dim,hidden_dim,config['output_dim_mapping']["right_elbow_orientation"])
      right_palm_position_list = right_palm_position(test_loader,input_dim,hidden_dim,config['output_dim_mapping']["right_palm_position"])
      right_fingers_joined_list = right_fingers_joined(test_loader,input_dim,hidden_dim,config['output_dim_mapping']["right_fingers_joined"])
      right_finger_orientation_list = right_finger_orientation(test_loader,input_dim,hidden_dim,config['output_dim_mapping']["right_finger_orientation"])
      right_finger_closeness_to_face_list = right_finger_closeness_to_face(test_loader,input_dim,hidden_dim,config['output_dim_mapping']["right_finger_closeness_to_face"])

      feature_vector =  right_arm_folded_list + right_forearm_orientation_list + right_hand_position_along_body_list + right_elbow_orientation_list + right_palm_position_list + right_fingers_joined_list + right_finger_orientation_list + right_finger_closeness_to_face_list
      print("feature_vector",feature_vector)
      vector_list.append(feature_vector)
    print("vector_list",len(vector_list))
    feature_list.append(vector_list)
  labels = []
  vid_name = []
  for vid in classification_dict.keys():
      labels.append(vid.split("_")[1])
      vid_name.append(vid)
  # Example DataFrame
  seq_ip_data = {'vid_name': vid_name,
          'feature_list': feature_list,
          'labels': labels}

  df_seq_ip = pd.DataFrame(seq_ip_data)

  df_seq_ip = pd.DataFrame(seq_ip_data)

  print(df_seq_ip)

  data = SequenceClassificationDataset(df_seq_ip)
  dataloader = torch.utils.data.DataLoader(data)


  label_mapping = {0: 'beautiful', 1: 'brush', 2: 'bye', 3: 'cook', 4: 'dead', 5: 'go', 6: 'good', 7: 'hello', 8: 'sorry', 9: 'thankyou'}


  input_size = 8
  hidden_size = 512
  num_layers = 3
  output_size = 10
  dropout = 0.2
  max_epoch = 50
  learning_rate = 0.0001

  model = Classifier(input_size, hidden_size, num_layers, output_size ,dropout)

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


# pred = ['beautiful', 'beautiful', 'beautiful', 'go', 'go', 'cook', 'go', 'go', 'bye', 'thankyou', 'thankyou', 'thankyou', 'sorry', 'sorry', 'sorry', 'dead', 'dead', 'dead', 'beautiful', 'beautiful', 'beautiful', 'hello', 'hello', 'hello', 'bye', 'bye', 'bye', 'go', 'go', 'go', 'beautiful', 'beautiful', 'beautiful', 'cook', 'go', 'cook', 'good', 'go', 'good', 'thankyou', 'thankyou', 'thankyou', 'sorry', 'sorry', 'sorry', 'dead', 'go', 'dead', 'brush', 'brush', 'brush', 'bye', 'bye', 'bye', 'bye', 'bye', 'bye', 'go', 'go', 'go']
# true = ['beautiful', 'beautiful', 'beautiful', 'cook', 'cook', 'cook', 'good', 'good', 'good', 'thankyou', 'thankyou', 'thankyou', 'sorry', 'sorry', 'sorry', 'dead', 'dead', 'dead', 'brush', 'brush', 'brush', 'hello', 'hello', 'hello', 'bye', 'bye', 'bye', 'go', 'go', 'go', 'beautiful', 'beautiful', 'beautiful', 'cook', 'cook', 'cook', 'good', 'good', 'good', 'thankyou', 'thankyou', 'thankyou', 'sorry', 'sorry', 'sorry', 'dead', 'dead', 'dead', 'brush', 'brush', 'brush', 'hello', 'hello', 'hello', 'bye', 'bye', 'bye', 'go', 'go', 'go']

# print(calculate_accuracy(pred,true))
# print(calculate_classwise_accuracy(pred,true))
