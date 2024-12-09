# from utility import construct_npy_dictionary,read_config

# if __name__ == '__main__':

#     npy_base_dir = '/4TBHD/ISL/CodeBase/Sequence_Model/v2_40sign_all_npyfolder'
#     frame_tmp_dir = '/4TBHD/ISL/CodeBase/Sequence_Model/tmp/frames_tmp'
#     config = read_config('config.yaml')

#     path_to_40_signs = '/4TBHD/ISL/data_preparation/seq_test_data'
#     construct_npy_dictionary(config,npy_base_dir,frame_tmp_dir,path_to_40_signs,npy_pickle_path = 'v2_40sign_all.pkl')




import pickle
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
from sklearn.utils import resample
from custom_dataloader import *
from utility import *
from sequence_classifier import *



parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(parent_dir)
sys.path.append(parent_dir)
from extract_non_blur_frames_server import *
from preprocess import *

def fit(model,loss_fn,optimizer,max_epoch,model_save_name):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # device = torch.device('cpu')
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    # Training loop with early stopping
    for epoch in range(max_epoch):
        model.train()  # Training mode
        total_loss = 0

        for X, Y in train_dataloader:
            X, Y = X.to(device), Y.to(device)

            pred = model(X)
            loss = loss_fn(pred, Y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        n = total_loss/len(train_dataloader)
        print(n)
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

    # Test accuracy (after training completes or early stopping triggers)
    test_acc ,_  = calculate_model_accuracy(model,loss_fn, test_dataloader,device)
    
    print(f'Test Accuracy: {test_acc:.2f}%')

    calculate_model_classwise_accuracy(model,loss_fn, test_dataloader,device,output_size)
    # Save the model
    # model_name = f"Model-HiddenSize-{hidden_size}_NumLayers-{num_layers}_Dropout-{dropout}"
    torch.save(model.state_dict(), f'/4TBHD/ISL/CodeBase/sign_models/{model_save_name}.h5')

    print('Training complete.')


# path_to_sequence_data = '/4TBHD/ISL/CodeBase/utils/scripts/v2_1_40_sign_all_comb_seq.pkl'


# if os.path.getsize(path_to_sequence_data) > 0:
#     with open(path_to_sequence_data, 'rb') as f:
#         classification_dict = pickle.load(f)
# else:
#     print("The file is empty.")

file_path_2 = '/4TBHD/ISL/CodeBase/Sequence_Model/test.pkl'

# The code below is used to read the pickle file and make the seq_data_pickle file
# dataset_size = "Expanded" # Reduced/Expanded
# construct_seq_dictionary(config,classification_dict,dataset_size,save_path=file_path_2)




if os.path.getsize(file_path_2) > 0:
    with open(file_path_2, 'rb') as f:
        seq_ip_data = pickle.load(f)
else:
    print("The file is empty.")

df_seq_ip = pd.DataFrame(seq_ip_data)

print(len(df_seq_ip))
# print(df_seq_ip.head())

# df_balanced = balance_classes(df_seq_ip,"labels")
# print(df_balanced)
train_list,val_list,test_list = [],[],[]

print(df_seq_ip['labels'].unique())
for label in df_seq_ip['labels'].unique():
    # Filter the dataframe for each label
    df_label = df_seq_ip[df_seq_ip['labels'] == label]


    print(df_label)
    
    # First, split into train and temp (test+val)
    df_train, df_temp = train_test_split(df_label, test_size=0.2, random_state=42, stratify=df_label['labels'])
    # print(df_temp)
    
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



# Comment the below when not doing sweep
input_size = 20
hidden_size = 256
num_layers = 1
output_size = 20
dropout = 0.5
max_epoch = 100
learning_rate = 0.0001

seq_model = 'LSTM'

model = Classifier(seq_model,input_size, hidden_size, num_layers, output_size ,dropout)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model_name = f'v2_40signs_LSTM_hidden-size:{hidden_size}_num-layers:{num_layers}'

fit(model,loss_fn,optimizer,max_epoch,model_name)

