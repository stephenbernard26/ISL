import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import pandas as pd
import os
import re
import yaml
import pandas as pd
import csv

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity 1
        self.relu1 = nn.ReLU()

        # Linear function 2: 100 --> 100
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()

        # Linear function 3: 100 --> 100
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 3
        self.relu3 = nn.ReLU()

        # Linear function 4 (readout): 100 --> 10
        self.fc4 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)

        # Linear function 2
        out = self.fc3(out)
        # Non-linearity 2
        out = self.relu3(out)

        # Linear function 4 (readout)
        out = self.fc4(out)
        return out


 
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
 
if __name__ == '__main__':

    config = read_config('config_train_inference.yaml')

 
    csv_path = config['inference']['csv_path']

    z,X_file_name = get_X_and_Y_value(csv_path)

    input_dim =  config['inference']["input_dim"]
    hidden_dim =  config['inference']["hidden_dim"]
    output_dim =  config['inference']["output_dim"]
    # num_epochs = 500

    model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load(config['inference']["checkpoint_path"]))


    # Convert to numpy arrays
    # data_x = np.array(data_x)  # shape (1000, 64, 2)
    # data_y = np.array(data_y)  # shape (1000, 19)

    data_z = np.array(z)  # shape (1000, 64, 2)

    # Flatten x to (num_samples, 128)
    data_z_flat = data_z.reshape(len(data_z), -1)

    # # Example data (replace with actual data)
    num_samples = len(data_z)
    batch_size =100

    # Convert to numpy arrays
    # data_x = np.array(data_x)  # shape (1000, 64, 2)
    # data_y = np.array(data_y)  # shape (1000, 6)

    # Convert to PyTorch tensors
    z_tensor = torch.tensor(data_z_flat, dtype=torch.float32)

    test_data = torch.utils.data.TensorDataset(z_tensor)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False)
    ### prediction ####
    predictions_dict = {}  # Dictionary to store inputs and their predictions
    total = 0
    # Ensure the model is in evaluation mode
    model.eval()

    # Iterate through test dataset
    for images in test_loader:
        # Extract the tensor from the list (since it's a list of inputs)
        images = images[0]  # Get the input tensor
        
        # Move images to the appropriate device (CPU/GPU)
        # images = images.to('cuda')  # If using GPU, otherwise omit or use .to('cpu')

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
    print(len(predictions_dict))

    # List of all inputs (keys in the dictionary)
    prediction_label_dict = {
        
        # right arm folded
        # 0:'partially folded',
        # 1: 'folded',
        # 2: 'rest',

        # right forearm orientation
        # 0:  "horizontal",
        # 1: "vertical",
        # 2: "lower diagonal",
        # 3: "upper diagonal",

        #  finger closeness to face
        0: "eyes",
        1: "ears",
        2: "nose",
        3: "lips",
        4: "chin",
        5: "cheeks",
        6: "none",
        7: "hair",
        8: "forehead"

    
        # # fingers joined
        # 0: 'joined',
        # 1: 'notjoined',
 
        #finger orientation
        # 0: 'open',
        # 1: 'closed',
        # 2: 'thumb alone open',
        # 3: 'index finger open',
        # 4: 'partially joined'
 
        #position along body
        # 0: 'chest',
        # 1: 'face',
        # 2: 'abdomen',
        # 3: 'belowabdomen'

        # right palm position
        # 0: "towards body",
        # 1: "upwards",
        # 2: "away from body",
        # 3: "downwards",
        # 4: "towards left",
        # 5: "towards right"
    }

    inputs_list = [s.replace('keypoints', 'onlyface').replace('.npy','.jpg') for s in X_file_name]
    predictions_list = list(predictions_dict.values())
    string_predictions_list = [prediction_label_dict[prediction] for prediction in predictions_list]

    # EXCEL results

    # !pip install pandas openpyxl


    # Assuming inputs_list and string_predictions_list are already populated

    # Convert inputs_list and predictions_list to a pandas DataFrame
    data = {
        "Input": [input_item.cpu().tolist() if isinstance(input_item, torch.Tensor) else str(input_item) for input_item in inputs_list],
        "Prediction": string_predictions_list
    }

    df = pd.DataFrame(data)

    os.makedirs('Test_Dataset_Prediction',exist_ok=True)
    # Path to save the Excel file
    excel_file_path = config['inference']['excel_file_path']
    
    # Save the DataFrame to an Excel file
    df.to_csv(excel_file_path, index=False)  # index=False to avoid writing row numbers

    # # Now, the Excel file contains the inputs and predictions
    print(f"Excel file saved at: {excel_file_path}")


