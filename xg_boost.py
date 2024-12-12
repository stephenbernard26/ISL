import numpy as np
import pickle

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def flatten(feature_list):

    vid_np = []
    for vid in feature_list:
        img_np = [np.load(img) for img in vid]  
        # print(np.array(img_np).shape)

        # Padding configuration: (pad_before, pad_after) for each dimension
        padding = ((0, desired_shape[0] - np.array(img_np).shape[0]),  # Pad along the first dimension
                (0, 0),  # No padding for the second dimension
                (0, 0))  # No padding for the third dimension

        # Pad the array with zeros
        padded_array = np.pad(np.array(img_np), padding, mode='constant', constant_values=0)
        padded_array = padded_array.reshape(-1,1)

        # print(padded_array)


        # break
        vid_np.append(padded_array)

    # print(vid_np.shape) 
    
    return vid_np

if __name__ == '__main__':


    # seq_dict_path = '/4TBHD/ISL/CodeBase/Sequence_Model/ai4b_50sign_no_threshold_det_trak_conf_0.2_reduced_sequence_vector.pkl'
    # seq_dict_path = "/4TBHD/ISL/CodeBase/Sequence_Model/pickle_files/5sign_5person_7video__vector_sequence_pkl/5sign_2video_5person.pkl"
    seq_dict_path = '/4TBHD/ISL/CodeBase/Sequence_Model/tmp_npy_folder/vid35_itel50/sandeep_dustbin_5/1.npy'
    with open(seq_dict_path, 'rb') as file:
        classification_dict = pickle.load(file)

    print((classification_dict.keys()))
    print(len(classification_dict))

    desired_shape = (200, 52, 2)


    label_mapping = {
                'dustbin': 0,   
                'coat': 1,

                'swimming': 2,
                'key': 3,

                'lock': 4
            }

    # label_mapping = {
    #                 'dustbin': 0,
    #                 'family': 1,
    #                 'tubelight': 2,
    #                 'earth': 3,
    #                 'hairband': 4,
    #                 'earmuffs': 5,
    #                 'below': 6,
    #                 'garden': 7,
    #                 'gift': 8,
    #                 'lock': 9,
    #                 'talk': 10,
    #                 'key': 11,
    #                 'conductor': 12,
    #                 'glasses': 13,
    #                 'lorry': 14,
    #                 'thursday': 15,
    #                 'bob': 16,
    #                 'party': 17,
    #                 'coat': 18,
    #                 'swimming': 19
    #             }



    X,y = [],[]
    for key,val in classification_dict.items():

        X.append(val)
        y.append(label_mapping[key.split('_')[1]])


    y = np.array(y, dtype=np.int32)


    X_train, X_tmp, y_train, y_tmp = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split( X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)


    print((y_train).shape)

    X_train = flatten(X_train)
    X_test = flatten(X_test)

    X_train = np.squeeze(X_train, axis=-1)
    X_test = np.squeeze(X_test, axis=-1)

    # print(type(X_train))
    print((np.array((X_train))).shape)

    # Convert data into DMatrix (XGBoost's internal data structure)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Define XGBoost parameters
    params = {
        'objective': 'multi:softmax',  # For multi-class classification
        'num_class': 5,               # Number of classes (for Iris dataset)
        'eval_metric': 'mlogloss',    # Evaluation metric
        'max_depth': 2,               # Maximum depth of trees
        'eta': 0.3,                   # Learning rate
        'seed': 42                    # Seed for reproducibility
    }

    # Train the XGBoost model
    num_round = 100
    bst = xgb.train(params, dtrain, num_round)

    # Make predictions
    predictions = bst.predict(dtest)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")
