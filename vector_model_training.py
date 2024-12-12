"""
Author: Stephen L

Code for Training the model

Input: Excel file which contains the path to npy file and the label

Modification :

    1. Added the code to run on GPU
    2. Added the Config file to read the configuration from one place



"""





import pandas as pd
import os
import re
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
import yaml


np.random.seed(42)




def split_filename(filename):
    # Regex to match a string followed by digits and ending with ".jpg"
    match = re.match(r"([a-zA-Z]+)(\d+)\.jpg", filename)
    
    if match:
        name_part = match.group(1)
        number_part = match.group(2)
        return name_part, number_part
    else:
        return None, None
    
group_value_mapping = {
 
    "rest": 2,
    "one": 1,
 
    'partially folded': 0,
    'folded': 1,
 
    "horizontal": 0,
    "vertical": 1,
    "lowerdiagonal": 2,
    "upperdiagonal": 3,
 
    "eyes": 0,
    "ears": 1,
    "nose": 2,
    "lips": 3,
    "chin": 4,
    "cheeks": 5,
    "none":6,
    "hair":7,
    "forehead":8,
 
    # 0: "eyes",
    # 1: "ears",
    # 2: "nose",
    # 3: "lips",
    # 4: "chin",
    # 5: "cheeks",
    # 6: "none",
    # 7: "hair",
    # 8: "forehead",

    'joined': 0,
    'not joined': 1,
 
    'open': 0,
    'closed': 1,
    'thumb alone open': 2,
    'index finger open': 3,
    'partially joined': 4,
 
    'chest': 0,
    'face': 1,
    'abdomen': 2,
    'belowabdomen':3,
 
    "towards body": 0,
    "upwards": 1,
    "away from body": 2,
    "downwards": 3,
    "towards left": 4,
    "towards right": 5  ,   
 
}



# def get_X_and_Y_value(csv_file_path,path_to_npy_files):
 
#     X_label,Y_label  = [],[]
 
#     data = pd.read_csv(csv_file_path)
 
#     print(data.columns)
#     for index, row in data.iterrows():
#         image_name = row[data.columns[0]]
#         orientation_key = row[data.columns[1]]
 
#         name_part, number_part = split_filename(image_name)
    
#         # npy_file_path = os.path.join(path_to_npy_files, f'{name_part}/{name_part}_{number_part}.npy')
#         npy_file_path = os.path.join(path_to_npy_files, f'{name_part}/{number_part}.npy')

#         X_label.append(np.load(npy_file_path))
#         Y_label.append(group_value_mapping[orientation_key])
 
        
 
#         print(image_name)
#         print(name_part+'_'+number_part+'.npy')
#         print(orientation_key)
    
#     return X_label,Y_label





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

# Function to test and train until correct prediction on one data point
def train_until_correct(model, val_loader, max_epochs=1000):
    # correct_prediction = False
    epoch = 0

    # Loop through the validation loader to get a batch
    for val_data, val_labels in val_loader:
        # Assuming we are dealing with a batch of data here
        for i in range(len(val_data)):
            correct_prediction = False
            single_data = val_data[i].unsqueeze(0)  # Get the first data point in the batch
            single_label = val_labels[i].unsqueeze(0)  # Get the first label in the batch
            print("single_label",single_label)

            while not correct_prediction and epoch < max_epochs:
                # Test the model on the single data point
                with torch.no_grad():
                    output = model(single_data)
                    _, predicted = torch.max(output.data, 1)
                    print(f'Epoch {epoch}: Predicted: {predicted.item()}, Actual: {single_label.item()}')

                if predicted.item() == single_label.item():
                    print(f"Model predicted correctly at epoch {epoch}.")
                    correct_prediction = True
                    print("epoch",epoch)
                else:
                    # Retrain the model on this single data point
                    print(f"Retraining: Model predicted incorrectly at epoch {epoch}.")
                    model.train()

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    output = model(single_data)

                    # Compute loss
                    loss = criterion(output, single_label)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    epoch += 1

            # if correct_prediction:
            #     break  # Exit the loop if the model predicted correctly

    if not correct_prediction:
        print(f"Model did not predict correctly after {max_epochs} epochs.")
    else:
        print(f"Training successful, model correctly predicted after {epoch} epochs.")

    return epoch

def validation_test(val_loader,model):
    correct = 0
    total = 0
    val_loss = 0
    # Iterate through test dataset
    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Load images with gradient accumulation capabilities
        # images = images.view(-1, 28*28).requires_grad_()

        # Forward pass only to get logits/output
        outputs = model(images)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        v_loss = criterion(outputs, labels)
        val_loss += v_loss.item()

        # Total number of labels
        total += labels.size(0)

        # Total correct predictions
        correct += (predicted == labels).sum()

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total

    # Print Loss
    print('Iteration: {}. Loss: {}. val Accuracy: {}'.format(iter, loss.item(), accuracy))

def test_data_test(test_loader,model):
    correct = 0
    total = 0
    test_loss = 0
    # Iterate through test dataset
    for images, labels in test_loader:


        images = images.to(device)
        labels = labels.to(device)
        # Load images with gradient accumulation capabilities
        # images = images.view(-1, 28*28).requires_grad_()

        # Forward pass only to get logits/output
        outputs = model(images)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        test_loss = criterion(outputs, labels)
        test_loss += test_loss.item()

        # Total number of labels
        total += labels.size(0)

        # Total correct predictions
        correct += (predicted == labels).sum()

    test_loss /= len(test_loader)
    accuracy = 100 * correct / total

    # Print Loss
    print('Iteration: {}. Loss: {}. Test Accuracy: {}'.format(iter, loss.item(), accuracy))


def read_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':


    if torch.cuda.is_available():  
        device = "cuda" 
    else:  
        device = "cpu"  

    print("device",device)

    # Reading the Config File
    config = read_config('config_train_inference.yaml')


    csv_path = config['training']['csv_path']
    path_to_npy_files = config['training']['path_to_npy_files']
    # path_to_npy_files = '/4TBHD/ISL/data_preparation/feature_keys/Keypoints/right_arm_folded/'

    data = pd.read_csv(csv_path)
    x,y=[],[]
    # print(data.columns)
    for index, row in data.iterrows():
        # print(index)
        image_name = row[data.columns[0]].replace(",","")
        orientation_key = row[data.columns[1]]
    
    # for i in range(len(path)):
    #     image_name = path[i]
    #     orientation_key = corrections[i]
        # print(orientation_key.replace(" ",""))
        # name_part, number_part = split_filename(image_name)
    
        # npy_file_path = os.path.join(path_to_npy_files, f'{name_part}/{name_part}_{number_part}.npy')
        npy_file_path = os.path.join(image_name.replace('onlyface','keypoints').replace('.jpg','.npy'))
    
        x.append(np.load(npy_file_path))
        y.append(group_value_mapping[orientation_key.replace(" ","")])




 
    # x,y = get_X_and_Y_value(csv_path,path_to_npy_files)
    print("number of classes",len(list(set(y))))
    
    data_x = np.array(x)  # shape (1000, 64, 2)
    data_y = np.array(y)  # shape

    # Flatten x to (num_samples, 128)
    data_x_flat = data_x.reshape(len(data_x), -1)

    # # Example data (replace with actual data)
    num_samples = len(data_x)
    batch_size = config['training']["batch_size"]

    # Convert to numpy arrays
    # data_x = np.array(data_x)  # shape (1000, 64, 2)
    # data_y = np.array(data_y)  # shape (1000, 6)

    # Convert to PyTorch tensors
    x_tensor = torch.tensor(data_x_flat, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Split data into train and test sets
    x_train, x_temp, y_train, y_temp = train_test_split(x_tensor, y_tensor, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # DataLoader
    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)

    val_data = torch.utils.data.TensorDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size, shuffle=False)

    test_data = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False)

    batch_size = config['training']["batch_size"]
    n_iters = config['training']['n_iters']
    num_epochs = n_iters / (len(train_data) / batch_size)

    input_dim = config['training']["input_dim"]
    hidden_dim = config['training']["hidden_dim"]
    output_dim = config['training']["output_dim"]
    num_epochs = config['training']["num_epochs"]

    model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    learning_rate = config['training']['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)

    iter = 0
    best_val_loss = float('inf')  # Initialize best validation loss to infinity
    patience = 10  # Number of epochs to wait before stopping if no improvement
    trigger_times = 0  # Counts how many times there was no improvement

    for epoch in range(num_epochs):
        correct = 0
        total = 0
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # print("images",images.shape)
            # print("labels",labels.shape)

            # Load images with gradient accumulation capabilities
            # images = images.view(-1, 28*28).requires_grad_()

            # # Clear gradients w.r.t. parameters
            # optimizer.zero_grad()

            # Forward pass to get output/logits

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Accumulate the loss
            running_loss += loss.item()

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += labels.size(0)

            # Total correct predictions
            correct += (predicted == labels).sum().item()

            iter += 1
        # Calculate training loss and accuracy
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total

        # Print Loss and Accuracy for each epoch
        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    

    #------------------------------------------------prof logic---------------------------------------------------
    # Set model to evaluation mode for testing
    # model.eval()
    # epoch = 51
    # while epoch > 0:
    #     # Call the function to test and retrain on the first data point
    #     epoch = train_until_correct(model, val_loader)
    #     print(epoch)

    model_name = config['training']['model_name']
    torch.save(model.state_dict(), f'/4TBHD/ISL/vector_models/10kmodels/{model_name}.h5')

    validation_test(val_loader,model)
    test_data_test(test_loader,model)
