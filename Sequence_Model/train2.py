"""
Author: Sandeep Kumar Suresh
        EE23S059

This code is used to train the sequence model .

Input : Directory which contains the videos for training

Output : Model Checkpoint

"""
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import random
import os
import numpy as np
from sklearn.model_selection import train_test_split
import sys

from inference import extract_npy_files
from custom_dataloader import *
from utility import *
from sequence_classifier import *



parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(parent_dir)
sys.path.append(parent_dir)
from extract_non_blur_frames_server import *
from preprocess import *
# from torchsummary import summary


# import wandb


# For Reproducibility
def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)



"""
# The below code activate the sweep mode for finding the best hyperparameters
# The below code was used to find the best hyperparameter configurations 

import wandb
sweep_configuration = {
    'method': 'bayes',  # 'grid' or 'random'
    'metric': {
        'name': 'validation_accuracy',
        'goal': 'maximize'   
    },
    'parameters': {
        'hidden_size': {  # LSTM hidden state size
            'values': [16,32,64,128, 256]
        },
        'num_layers': {  # Number of stacked LSTM layers
            'values': [1, 2, 3]
        },
        'dropout': {  # Dropout rate to prevent overfitting
            'values': [0.2, 0.3, 0.5]
        },
        'learning_rate': {  # Learning rate for optimizer
            'values': [1e-4, 1e-3]
        },
        'num_epoch':{
            'values':[50,100]
        },
        'model_name':{
            'values':['LSTM','BiLSTM']
        }
    }
}


def do_sweep():

    wandb.init()
    config = wandb.config
    run_name = (
        " | Hidden Size:" + str(config.hidden_size) +
        " | Num Layers:" + str(config.num_layers) +
        " | Dropout:" + str(config.dropout) 
        )
    print(run_name)
    wandb.run.name = run_name

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    input_size = 20
    hidden_size = config.hidden_size
    num_layers = config.num_layers
    output_size = 50
    dropout = config.dropout
    model_name = config.model_name

    model = Classifier(model_name,input_size, hidden_size, num_layers, output_size ,dropout)
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,weight_decay=1e-5)

    model.train()

    # Training loop
    for epoch in range(config.num_epoch):
        model.train()  # Training mode
        total_loss = 0

        for X, Y in train_dataloader:
            X = X.to(device)
            Y = Y.to(device)

            pred = model(X)
            loss = loss_fn(pred, Y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print epoch loss
        print(f'Epoch [{epoch+1}/{config.num_epoch}], Loss: {total_loss:.4f}')


        # Training accuracy
        train_acc , _ = calculate_model_accuracy(model,loss_fn,train_dataloader,device)
        print(f'Training Accuracy: {train_acc:.2f}%')

        # Validation accuracy
        val_acc , val_loss = calculate_model_accuracy(model,loss_fn,val_dataloader,device)
        print(f'Validation Accuracy: {val_acc:.2f}%')

        wandb.log({'training_loss':round(total_loss, 4) , "training_accuracy": train_acc , "validation_accuracy":val_acc })
        print(f'training_loss: {round(total_loss, 4)}  training_accuracy: '+f'{train_acc}  '+ f'validation_accuracy: {val_acc}\n')

        # Check early stopping condition
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    # Test accuracy (after training completes)
    test_acc , test_loss = calculate_model_accuracy(model,loss_fn,test_dataloader,device)
    print(f'Test Accuracy: {test_acc:.2f}%')

    print('Training complete.')
"""
def fit(model,loss_fn,optimizer,max_epoch,model_save_name):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    # Training loop with early stopping
    for epoch in range(max_epoch):
        model.train()  # Training mode
        total_loss = 0

        for X, Y in train_dataloader:
            # print("X",X)
            # print("Y",Y)
            X, Y = X.to(device), Y.to(device)

            pred = model(X)
            loss = loss_fn(pred, Y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        n = total_loss/len(train_dataloader)
        # print(n)
        # Print epoch loss
        print(f'Epoch [{epoch+1}/{max_epoch}], Training Loss: {total_loss:.4f}')

        # # Calculate validation loss
        # model.eval()  # Evaluation mode
        # val_loss = 0
        # with torch.no_grad():
        #     for X_val, Y_val in val_dataloader:
        #         X_val, Y_val = X_val.to(device), Y_val.to(device)
        #         pred_val = model(X_val)
        #         loss_val = loss_fn(pred_val, Y_val)
        #         val_loss += loss_val.item()

        # val_loss /= len(val_dataloader)
        # print("val_loss1",val_loss)

        val_acc,val_loss = calculate_model_accuracy(model, loss_fn,val_dataloader,device)
        print("val_loss2",val_loss)
        print(f'Validation Loss: {val_loss:.4f}')

        # Validation accuracy
        print(f'Validation Accuracy: {val_acc:.2f}%')

        # Check early stopping condition
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # # Test accuracy (after training completes or early stopping triggers)
    # test_acc ,_  = calculate_model_accuracy(model,loss_fn, test_dataloader,device)
    
    # print(f'Test Accuracy: {test_acc:.2f}%')

    # calculate_model_classwise_accuracy(model,loss_fn, test_dataloader,device,output_size)
    # # Save the model
    # # model_name = f"Model-HiddenSize-{hidden_size}_NumLayers-{num_layers}_Dropout-{dropout}"
    # torch.save(model.state_dict(), f'/4TBHD/ISL/CodeBase/sign_models/{model_save_name}.h5')

    # print('Training complete.')

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return torch.utils.data.default_collate(batch)



if __name__ == '__main__':

    # Checking the disk where to run the code

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    set_random_seed(42)

    video_dataset_path = '/4TBHD/ai4/ai4bharat_portrait_dataset/dataset'
    experiment_name = 'include_260'
    # npy_dict = {}
    # # npy_dict = defaultdict(list)
    npy_base_dir = f"/4TBHD/ISL/CodeBase/Sequence_Model/tmp_npy_folder/{experiment_name}"
    frame_tmp_dir = '/4TBHD/ISL/CodeBase/Sequence_Model/tmp/frames_tmp'
    seq_npy_dict = f'/4TBHD/ISL/CodeBase/Sequence_Model/{experiment_name}.pkl'
    os.makedirs(npy_base_dir,exist_ok=True)

    config = read_config('config.yaml')



    # if os.path.exists(seq_npy_dict):
    #     print("File exists, skipping construct_npy_dictionary")

    # else:        
    #     construct_npy_dictionary(config,npy_base_dir,video_dataset_path,seq_npy_dict)

    # path_to_sequence_data = seq_npy_dict


    # Uncomment the below if you want to run for a specific model
    # path_to_sequence_data = '/4TBHD/ISL/CodeBase/Sequence_Model/include_260_aug.pkl'

    # if os.path.getsize(path_to_sequence_data) > 0:
    #     with open(path_to_sequence_data, 'rb') as f:
    #         classification_dict = pickle.load(f)
    # else:
    #     print("The file is empty.")
    
    # print((classification_dict["cindrella_thursday_4"]))

    # print((classification_dict["cindrella_thursday_4_plus7rotation"].shape))

    # print((classification_dict["cindrella_thursday_4"].shape))

    # print(type(classification_dict["cindrella_thursday_4_plus7rotation"]))



    # file_path_2 = f'/4TBHD/ISL/CodeBase/Sequence_Model/{experiment_name}_reduced_sequence_vector.pkl'

    file_path_2 = '/4TBHD/ISL/CodeBase/Sequence_Model/pickle_files/include/include_260_sequence_vector.pkl'


    # if os.path.exists(file_path_2):
    #     print("File exists, skipping construct_seq_dictionary")
    # else:       
    #     dataset_size = "reduced" # reduced/Expanded
    #     shift_orgin = False
    #     construct_seq_dictionary(config,classification_dict,dataset_size,shift_orgin,save_path=file_path_2)

    if os.path.getsize(file_path_2) > 0:
        with open(file_path_2, 'rb') as f:
            seq_ip_data = pickle.load(f)
    else:
        print("The file is empty.")

    path_to_sequence_vector = file_path_2
    # path_to_sequence_vector = '/4TBHD/ISL/CodeBase/Sequence_Model/pickle_files/last_20_sign_vector_sequence_pkl/last_20_seq_vector_reduced.pkl'
    # path_to_sequence_vector = '/4TBHD/ISL/CodeBase/Sequence_Model/pickle_files/30_sign_vector_sequence_pkl/30_sign_seq_vector.pkl'


    df = pd.read_pickle(path_to_sequence_vector)
    df_seq_ip = pd.DataFrame(df)

    print('column names',df_seq_ip.columns)



    # labels = df["labels"]
    # features = df["feature_list"]


    X = df_seq_ip['feature_list']  # Selecting the input features
    y = df_seq_ip['labels']                  # Target labels
    if any(element is None for element in X):
        print("The list contains NoneType elements.")
    else:
        print("The list does not contain any NoneType elements.")
    # X_train, X_tmp, y_train, y_tmp = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)
    # X_val, X_test, y_val, y_test = train_test_split( X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)


    # The below code is created when the split was uneven and could not be possible

    X_train, X_tmp, y_train, y_tmp = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)
    # X_val, X_test, y_val, y_test = train_test_split( X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)




    # print(X_train)
    # print(X_val)
    # print(X_test)

    # from itertools import zip_longest

    # with open('data_split.txt', 'w') as f:
    #     for a, b, c in zip_longest(X_train, X_val, X_test, fillvalue=None):
    #         f.write(f"{a}, {b}, {c}\n")


    # train, tmp = train_test_split(df_seq_ip, test_size=0.2,random_state=42)
    # val,test = train_test_split(tmp,test_size=0.5,random_state=42)
    # train_dataset = SequenceClassificationDataset(train)
    # val_dataset = SequenceClassificationDataset(val)
    # test_dataset = SequenceClassificationDataset(test)



    # test_df = pd.concat(test_list).reset_index(drop=True)






    # train_dataset = SequenceClassificationDataset(X_train,y_train)
    # val_dataset = SequenceClassificationDataset(X_val,y_val)
    # test_dataset = SequenceClassificationDataset(X_test,y_test)


    # The below code for uneven split


    print("invalid label", X_train[1084])

    train_dataset = SequenceClassificationDataset(X_train,y_train)
    val_dataset = SequenceClassificationDataset(X_tmp,y_tmp)



    train_dataloader = torch.utils.data.DataLoader(train_dataset,t)
    val_dataloader = torch.utils.data.DataLoader(val_dataset)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset)






    # # Comment the below when doing the sweep
    # sweep_id = wandb.sweep(sweep_configuration,project='sequence_classifier')
    # wandb.agent(sweep_id ,function=do_sweep,count=100)
    # wandb.finish()


    # Comment the below when not doing sweep
    input_size = 20
    hidden_size = 256
    num_layers = 1
    output_size = 262
    dropout = 0.2
    max_epoch = 1000
    learning_rate = 0.0001
    
    seq_model = 'BiLSTM'

    model = Classifier(seq_model,input_size, hidden_size, num_layers, output_size ,dropout)
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model_name = f'{experiment_name}_LSTM_hidden-size:{hidden_size}_num-layers:{num_layers}'
    
    fit(model,loss_fn,optimizer,max_epoch,model_name)

