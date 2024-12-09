import pickle
import pandas as pd
from utility import *
def construct_npy_dictionary(config,npy_base_dir,frame_tmp_dir,test_data_path,npy_pickle_path = 'npy_dict_all.pkl'):

    folders_to_extract = ['anirudh']
    videos_to_extract = ['below']

    npy_dict = {}
    for Signers in os.listdir(test_data_path):
        if Signers in folders_to_extract:
            for folders in os.listdir(os.path.join(test_data_path,Signers)):
                if folders in videos_to_extract:
                    for video in os.listdir(os.path.join(test_data_path,Signers,folders)):
                        video_path = os.path.join(test_data_path,Signers,folders,video)
            #             # print(video_path)
                        keyframes = extract_start_end_frames(video_path)
            #             # print(keyframes)
                        clear_tmp_directory(directory_path=frame_tmp_dir)
                        extract_seq_of_frames(video_path ,keyframes,frame_tmp_dir)
                    
                        for img in sorted(os.listdir(frame_tmp_dir),key=lambda x: int(x.split('.')[0])):
                            video_name = video
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


if __name__ == '__main__':


    path_to_video = '/4TBHD/ISL/data_preparation/v2_40sign_videos'

    npy_base_dir = "/4TBHD/ISL/CodeBase/Sequence_Model/anirudh_npy_folder"
    frame_tmp_dir = '/4TBHD/ISL/CodeBase/Sequence_Model/tmp/frames_tmp'
    seq_npy_dict = 'anirudh_npy_dict.pkl'
    os.makedirs(npy_base_dir,exist_ok=True)


    config = read_config('config.yaml')



    # Comment the below code if your dataset is already converted to pickle file
    construct_npy_dictionary(config,npy_base_dir,frame_tmp_dir,path_to_video,seq_npy_dict)


    path_to_sequence_data = seq_npy_dict


    if os.path.getsize(path_to_sequence_data) > 0:
        with open(path_to_sequence_data, 'rb') as f:
            classification_dict = pickle.load(f)
    else:
        print("The file is empty.")


    file_path_2 = f'anirudh_npy_dict_sequence_vector.pkl'

    # The code below is used to read the pickle file and make the seq_data_pickle file
    dataset_size = "Expanded" # Reduced/Expanded
    construct_seq_dictionary(config,classification_dict,dataset_size,save_path=file_path_2)


    # # path = '/4TBHD/ISL/CodeBase/Sequence_Model/intern_sign_all_seq_vector.pkl'
    # path = 'test.pkl'


    # with open(path, 'rb') as f:
    #     x = pickle.load(f)
    
    # df_seq_ip = pd.DataFrame(x)
    # print(df_seq_ip)

    # # Update label using replace method
    # df_seq_ip['labels'] = df_seq_ip['labels'].replace('good', 'goodmorning')

    # with open('test.pkl', 'wb') as f:
    #     pickle.dump(df_seq_ip,f)
    


    # label_to_find = 'aeroplane'
    # filtered_df = df_seq_ip[df_seq_ip['labels'] == label_to_find]

    # # Display vid_name column
    # print(filtered_df['vid_name'].tolist())

    # unique_labels = df_seq_ip['labels'].unique()
    # print(unique_labels)


    # with open(path, 'rb') as f:
    #     x = pickle.load(f)

    # print(x)
    # df_seq_ip = pd.DataFrame(x)

    # print(df_seq_ip)


    # path = '/4TBHD/ISL/CodeBase/Sequence_Model/pickle_files/last_20_sign_vector_sequence_pkl/last_20_seq_vector_reduced.pkl'

    # with open(path, 'rb') as f:
    #     x = pickle.load(f)
    
    
    # for k,v in x.items():
    #     print(k)
    # print(type(x))

    # print(x)

    # df_seq_ip = pd.DataFrame(x)

    # unique_labels = df_seq_ip['labels'].unique()
    # print(unique_labels)

    # label_to_find = 'bye'
    # filtered_df = df_seq_ip[df_seq_ip['labels'] == label_to_find]

    # print(filtered_df['vid_name'])

    # df_seq_ip_x = pd.DataFrame(x)
    # print(len(df_seq_ip_x))

    # path = '/4TBHD/ISL/CodeBase/Sequence_Model/v1_seq_vector(20_feat).pkl'

    # with open(path, 'rb') as f:
    #     y = pickle.load(f)
        
    # df_seq_ip_y = pd.DataFrame(y)

    # print(len(df_seq_ip_y))

    # # print(df_seq_ip_y)

    # result = pd.concat([df_seq_ip_y, df_seq_ip_x], ignore_index=True)

    # print(result)

    # unique_labels = result['labels'].unique()
    # print((unique_labels))

    # with open('30_sign_seq_vector.pkl', 'wb') as f:
    #     pickle.dump(result,f)