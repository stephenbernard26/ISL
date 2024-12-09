import pandas as pd
import os
import re
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.nn as nn
import random


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_prob):
        """
        input_dim: Size of the input layer.
        hidden_dims: A list of integers where each integer specifies the number of neurons in that hidden layer.
        output_dim: Size of the output layer.
        dropout_prob: Dropout probability.
        """
        super(FeedforwardNeuralNetModel, self).__init__()
        
        # Create lists to hold the layers and dropouts dynamically
        self.fc_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # First hidden layer (input_dim -> first hidden layer size)
        self.fc_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.relu_layers.append(nn.ReLU())
        self.dropout_layers.append(nn.Dropout(p=dropout_prob))

        # Create hidden layers dynamically (hidden_dims[i-1] -> hidden_dims[i])
        for i in range(1, len(hidden_dims)):
            self.fc_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.relu_layers.append(nn.ReLU())
            self.dropout_layers.append(nn.Dropout(p=dropout_prob))

        # Output layer (last hidden layer -> output_dim)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        # Apply fixed weight initialization
        self.apply(self.initialize_weights)

    def forward(self, x):
        # Pass through each hidden layer with activation and dropout
        for fc, relu, dropout in zip(self.fc_layers, self.relu_layers, self.dropout_layers):
            x = fc(x)
            x = relu(x)
            x = dropout(x)

        # Output layer (no activation here, since it could be for classification or regression)
        x = self.output_layer(x)
        return x

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            # Initialize weights with Xavier uniform
            torch.nn.init.xavier_uniform_(layer.weight)
            # Initialize biases with zeros
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
    np.random.seed(random_seed)
    random.seed(random_seed)

def data_preparation(data,vector_config,config,file_name):
    set_random_seed(42)
    x,y=[],[]
    print(data.columns)
    for index, row in data.iterrows():
        # print(index)
        image_name = row[data.columns[0]].replace(",","")
        orientation_key = row[data.columns[1]]

        npy_file_path = os.path.join(image_name.replace('onlyface','keypoints').replace('.jpg','.npy'))
        features = np.load(npy_file_path)

        selected_features = []
        for key in config['feature_indices'][file_name]:
            # Construct the filter key from `filter_mapping`
            filter_key = f"{key}_filter"
            # Only add mapped values if the filter key exists in `filter_mapping`
            if filter_key in config['filter_mapping']:
                for item in config['feature_indices'][file_name][key]:
                    # Append mapped value if it exists in the filter
                    if item in config['filter_mapping'][filter_key]:
                        selected_features.append(config['filter_mapping'][filter_key][item])
        
        x.append(features[selected_features])
        # x.append(np.load(npy_file_path))
        y.append(vector_config['encoder'][file_name][orientation_key.replace(" ","_")])
    
    print("number of classes --> ",len(list(set(y))))
    print("number of keypoints to be used -->",len(x[0]))

    data_x = np.array(x)  # shape (1000, 64, 2)
    data_y = np.array(y)  # shape

    # Flatten x to (num_samples, 128)
    data_x_flat = data_x.reshape(len(data_x), -1)

    # # Example data (replace with actual data)
    num_samples = len(data_x)
    batch_size =100

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

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    file_name = 'left_fingertips_orientation'
    # data = pd.read_csv(f'/4TBHD/ISL/CodeBase/Dataset/right_hand_dataset/Train/shifted_origin_v2_post4.7k/{file_name}.csv')
    data = pd.read_csv(f'/4TBHD/Janaghan/normalized/left_hand/{file_name}.csv')
    with open('/4TBHD/ISL/CodeBase/Vector_Model/label_mapping.yaml', 'r') as file:
            vector_config = yaml.safe_load(file)
    with open('/4TBHD/ISL/CodeBase/Vector_Model/config.yaml', 'r') as file:
            config = yaml.safe_load(file)

    if torch.cuda.is_available():  
        device = "cuda" 
    else:  
        device = "cpu"  

    train_loader, val_loader, test_loader = data_preparation(data,vector_config,config,file_name)

    model = FeedforwardNeuralNetModel(vector_config['input_dim_mapping'][file_name]*2, vector_config['hidden_dim_mapping'][file_name], vector_config['output_dim_mapping'][file_name], vector_config['dropout_prob'])
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-5)

    num_epochs = 1000
    iter = 0
    best_val_loss = float('inf')  # Initialize best validation loss to infinity
    patience = 10  # Number of epochs to wait before stopping if no improvement
    trigger_times = 0  # Counts how many times there was no improvement
    loss_list=[]
    val_loss_list = []

    for epoch in range(num_epochs):
        correct = 0
        total = 0
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            # print("images",images)
            # print("labels",labels)

            # Load images with gradient accumulation capabilities
            # images = images.view(-1, 28*28).requires_grad_()

            # # Clear gradients w.r.t. parameters
            # optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass to get output/logits
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
        loss_list.append(avg_loss)
        
        # Print Loss and Accuracy for each epoch
        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # Calculate validation loss
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)

        # Print validation loss
        print(f'Validation Loss: {avg_val_loss:.4f}')

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved at epoch {epoch+1}')
        else:
            trigger_times += 1
            print(f'No improvement for {trigger_times} epoch(s)')

        if trigger_times >= patience:
            print(f'Early stopping triggered after {trigger_times} epochs.')
            break

    # Calculate Accuracy         
    correct = 0
    total = 0
    val_loss = 0
    # Iterate through test dataset
    for images, labels in val_loader:
        # Forward pass only to get logits/output
        images = images.to(device)
        labels = labels.to(device)
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
    print('validation : Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

    correct = 0
    total = 0
    test_loss = 0
    # Iterate through test dataset
    for images, labels in test_loader:
        # Load images with gradient accumulation capabilities
        # images = images.view(-1, 28*28).requires_grad_()

        # Forward pass only to get logits/output
        images = images.to(device)
        labels = labels.to(device)
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
    print('Test: Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

    # torch.save(model.state_dict(), f'/4TBHD/ISL/CodeBase/vector_models/two_hands_v1/{file_name}.h5')
