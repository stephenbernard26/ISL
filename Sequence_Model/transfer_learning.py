"""
Author : Sandeep Kumar Sures
        EE23S059

"""

import torch
from model import *
import pandas as pd
from sklearn.model_selection import train_test_split
from custom_dataloader import *
from utility import *
from sequence_classifier import EarlyStopping
from collections import Counter
import random

# For Reproducibility
random.seed(0)
np.random.seed(0)


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


def fit(new_model,num_epochs,train_dataloader,val_dataloader,test_dataloader,optimizer,criterion,device,model_save_name):

    for epoch in range(num_epochs):
        new_model.train()
        total_loss = 0.0

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = new_model(inputs)  # Pass through the new layer
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_acc, train_loss = calculate_model_accuracy(new_model,criterion,train_dataloader,device)
        val_acc,val_loss = calculate_model_accuracy(new_model, criterion,val_dataloader,device)

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss / len(train_dataloader):.4f}, '
            f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}, '
            f'Training Accuracy: {train_acc:.2f}')

        # Check early stopping condition
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    test_acc,test_loss = calculate_model_accuracy(new_model,criterion,test_dataloader,device)
    print(f'Testing_acc : {test_acc}')

    calculate_model_classwise_accuracy(new_model,criterion, test_dataloader,device,50)

    torch.save(new_model.state_dict(), model_save_name)


def inference(new_model,test_dataloader,optimizer,criterion,device):


    label_mapping = {0: 'beautiful', 1: 'brush', 2: 'bye', 3: 'cook', 4: 'dead', 5: 'go', 6: 'good', 7: 'hello', 8: 'sorry', 9: 'thankyou'}

    new_model.eval()
    
    predictions,true_labels = [],[]
    with torch.no_grad():
        for X,Y in test_dataloader:
            X, Y = X.to(device), Y.to(device)

            pred = new_model(X)
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



if __name__ == '__main__':

    config = read_config('config.yaml')

    checkpoint_path = '/4TBHD/ISL/CodeBase/sign_models/augs_transformer.pth.small'

    dataset_type = 'reduced'

    # file_path = 'npy_dict_all.pkl'
    # if os.path.getsize(file_path) > 0:
    #     with open(file_path, 'rb') as f:
    #         classification_dict = pickle.load(f)
    # else:
    #     print("The file is empty.")
    # construct_seq_dictionary(config,classification_dict,dataset_type)



    path_to_sequence_data = f'/4TBHD/ISL/CodeBase/Sequence_Model/seq_train_data_{dataset_type}_v3.pkl'
    model_save_name = f'/4TBHD/ISL/CodeBase/sign_models/Transformer_small_finetuned_{dataset_type}_v3.h5'
    # test_dataset_path = 'seq_ip_data.pkl'
    test_dataset_path = 'v2_40sign_all_mod_comb_seqvector.pkl'
    model_load_path = f"/4TBHD/ISL/CodeBase/sign_models/Transformer_small_finetuned_{dataset_type}_v3.h5"

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(device)

    early_stopping = EarlyStopping(patience=5, min_delta=0.001)


    # Loading the train and test Dataloader


    # df = pd.read_pickle(path_to_sequence_data)
    # df = pd.DataFrame(df)

    # print(df)

    # labels = df["labels"]
    # features = df["feature_list"]





    # train, tmp = train_test_split(df, test_size=0.2,random_state=42)
    # val,test = train_test_split(tmp,test_size=0.5,random_state=42)


    # train_dataset = SequenceClassificationDataset(train)
    # val_dataset = SequenceClassificationDataset(val)
    # test_dataset = SequenceClassificationDataset(test)


    # train_dataloader = torch.utils.data.DataLoader(train_dataset)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset)


    # Loading the Transformer Configurations


    # pretrained_model().to(device)


    # Load the transformer model

    if checkpoint_path.split('/')[-1].split('.')[-1] == 'small':

        config = TransformerConfig(size='small')
    
    else:
        config = TransformerConfig(size='large')



    pretrained_model = Transformer(config)


    num_class = 50
    in_feat = 16

    # Load the custom model checkpoint (.pth)
    pretrained_in_feat = 134
    pretrained_out_feat = 50
    state_dict = torch.load(checkpoint_path) 
    model_state_dict = state_dict['model']
    pretrained_model.load_state_dict(model_state_dict,strict = False)


    projection_layer = nn.Linear(in_feat,pretrained_in_feat)
    new_layer = nn.Linear(pretrained_out_feat,num_class)

    # Defining a New Model that customize to our requirement




    new_model = Custom_Model(pretrained_model,projection_layer,new_layer).to(device)

    
    for params in pretrained_model.parameters():
        params.requires_grad = False

    print(new_model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)


    # Training Loop
    num_epochs = 100





    if os.path.getsize(test_dataset_path) > 0:
        with open(test_dataset_path, 'rb') as f:
            seq_ip_data = pickle.load(f)
    else:
        print("The file is empty.")

    df_seq_ip = pd.DataFrame(seq_ip_data)



    df_balanced = balance_classes(df_seq_ip,"labels")
    print(df_balanced)
    train_list,val_list,test_list = [],[],[]


    for label in df_balanced['labels'].unique():
        # Filter the dataframe for each label
        df_label = df_balanced[df_balanced['labels'] == label]

        # print(df_label)
        
        # First, split into train and temp (test+val)
        df_train, df_temp = train_test_split(df_label, test_size=0.2, random_state=42, stratify=df_label['labels'])
        
        # Then, split temp into validation and test
        df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['labels'])
        
        # Append to respective lists
        train_list.append(df_train)
        val_list.append(df_val)
        test_list.append(df_test)


    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)


    train_dataset = SequenceClassificationDataset(train_df)
    val_dataset = SequenceClassificationDataset(val_df)
    test_dataset = SequenceClassificationDataset(test_df)


    train_dataloader = torch.utils.data.DataLoader(train_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset)



    fit(new_model,num_epochs,train_dataloader,val_dataloader,test_dataloader,optimizer,criterion,device,model_save_name)







    # data_test = SequenceClassificationDataset(df_seq_ip)
    # dataloader_test = torch.utils.data.DataLoader(data_test)

    # model_test = Custom_Model(pretrained_model,projection_layer,new_layer)
    # model_test.load_state_dict(torch.load(model_load_path))

    # inference(model_test,dataloader_test,optimizer,criterion,device)
    

 

    # Save the fine-tuned model
    # torch.save(model.state_dict(), 'fine_tuned_model.pth')







    # # Freezing the model parameters and unfreezing the last layer

    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.new_layer.parameters():
    #     param.requires_grad = True

