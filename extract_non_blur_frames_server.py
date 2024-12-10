"""
Author: Sandeep Kumar Suresh
        EE23S059

Code to extract non-blured frames into a csv file

NOTE : 

    1. Remember to clear the non_blur_frames.csv file since it is in append mode
    2. The video in the path '/4TBHD/ISL/SIGN_LANG_AUG/Venkatesh/Good/venkatesh_good_19.mp4' had null keypoint
        when threshold was set to 5 . Reducing the threshold to 4 gave first and last frames
    3. The video in the above path was manually updated using the code snippet given in the end

Modifications:

    1. Added ordering while doing os loop
    2. Added getting only the non-blur frames
    3. Added left hand and right hand non-blur frames check
    
"""

import os 
import numpy as np
import cv2
import csv
from tqdm import tqdm

def clear_tmp_directory(directory_path):
    """
    Delete all files in the given directory using the os module.
    """
    try:
        # List all files in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            # Ensure only files are removed, not subdirectories
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                print(f"Skipping non-file item: {file_path}")
    except Exception as e:
        print(f"Error clearing directory {directory_path}: {e}")


def check_non_blur_frames(npy_dir_path,frame_no_path):
    """
    Code to check whether the frames are blurry or not

    Input: frame
    Output: True or False
    """
    npy_load = np.load(os.path.join(npy_dir_path,frame_no_path))
    # print(npy_load)
    if np.sum(npy_load[-4]) != 0.0:
        return True
    else:
        return False

def extract_start_end_frames(video_path, threshold=5):
    """
    This functions extracts the start and end of a keyframe
    """

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return []
    
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    key_frames = [] 
    
    frame_count = 1
    
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        frame_diff = cv2.absdiff(prev_frame_gray, curr_frame_gray)
        
        diff_mean = np.mean(frame_diff)
        
        if diff_mean > threshold:
            key_frames.append(frame_count)
            prev_frame_gray = curr_frame_gray

            # cv2.imwrite(f'tmp_dir/frame_{frame_count}.jpg', curr_frame)

        
        frame_count += 1
    

    cap.release()

    if not key_frames:
        threshold -= 1


    return key_frames

def extract_start_end_frames_with_decrementing_threshold_function(video_path, initial_threshold=0, min_threshold=0):
    threshold = initial_threshold
    while threshold >= min_threshold:
        key_frames = extract_start_end_frames(video_path, threshold)
        if len(key_frames) > 2:
            return key_frames  # Return key frames if found
        else:
            print(f"No key frames found at threshold {threshold}. Decreasing threshold...")
            threshold -= 1  # Decrease threshold if no frames found

    print("No key frames detected even at minimum threshold.")
    return []


def extract_seq_of_frames(config,video_path  ,npy_save_dir,lock,video_name,npy_dict):
    """
    This function extracts the sequence of frames from start and 
    the end of the keyframe
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # if frame_count >= key_frames[0] and frame_count <= key_frames[-1]:

        # Instead of saving , directly giving to extract npy function
        frame_no = extract_npy_files(config,frame,npy_save_dir,frame_count)

        if frame_no:
            with lock:
                if video_name not in npy_dict:
                    npy_dict[video_name] = []
                npy_dict[video_name].append(f'{npy_save_dir}/{frame_no}.npy')

        # if frame_count > key_frames[-1]:
        #     break
        frame_count += 1

    # Release the video capture object
    cap.release()


if __name__ == '__main__':
    videopath = '/4TBHD/ISL/data_preparation/v2_40sign_videos'

    frame_dir = '/4TBHD/ISL/data_preparation/v2_40sign_videos_kpt_face/onlyface'

    npy_dir = '/4TBHD/ISL/data_preparation/v2_40sign_videos_kpt_face/keypoints'

    csv_filename = 'v2_40signs_non_blur_frames.csv'

    frame_list = []

    for folders in tqdm(os.listdir(videopath)):

        if folders != "Test" and folders != "Hari"  :

            for sign_folder in os.listdir(os.path.join(videopath,folders)):
                for videos in os.listdir(os.path.join(videopath,folders,sign_folder)):
                    video_name = videos.split('.')[0]
                    # print(video_name)
                    keyframes = extract_start_end_frames(os.path.join(videopath,folders,sign_folder,videos))

                    for img_name in sorted(os.listdir(os.path.join(frame_dir,folders,sign_folder,video_name)),key=lambda x: int(x.split('.')[0])):

                        
                        # if int(img_name.split('.')[0]) > keyframes[0] and int(img_name.split('.')[0]) < keyframes[-1]:
                            

                        #     with open(csv_filename, 'a', newline='') as csvfile:
                        #         csvwriter = csv.writer(csvfile)
                        #         csvwriter.writerow([os.path.join(frame_dir,folders,sign_folder,video_name,img_name)])
                        try:
                            if len(keyframes) > 0:  # Check if keyframes list is not empty
                                if int(img_name.split('.')[0]) > keyframes[0] and int(img_name.split('.')[0]) < keyframes[-1]:
                                    npy_load = np.load(os.path.join(npy_dir,folders,sign_folder,video_name,img_name).replace('jpg','npy'))
                                    # print(npy_load)
                                    if np.sum(npy_load[-4]) and np.sum(npy_load[32:35]) != 0.0:                                    
                                        with open(csv_filename, 'a', newline='') as csvfile:
                                            csvwriter = csv.writer(csvfile)
                                            csvwriter.writerow([os.path.join(frame_dir, folders, sign_folder, video_name, img_name)])
                            else:
                                raise ValueError("Keyframes list is empty")
                        except (IndexError, ValueError) as e:
                            print(f"Error with video: {video_name} - {str(e)}")


        else:
            print("sfs")




            # break

    # test_video = '/4TBHD/ISL/SIGN_LANG_AUG/Venkatesh/Good/venkatesh_good_19.mp4'
    # keyframes = extract_start_end_frames(test_video)
    # print(keyframes)

    # for img_name in os.listdir('/4TBHD/ISL/data_preparation/test_all/keypoints/Venkatesh/Good/venkatesh_good_19'):


    #     if int(img_name.split('.')[0]) > keyframes[0] and int(img_name.split('.')[0]) < keyframes[-1]:
    

    #         with open(csv_filename, 'a', newline='') as csvfile:
    #             csvwriter = csv.writer(csvfile)
    #             csvwriter.writerow([f"/4TBHD/ISL/data_preparation/test_all/keypoints/Venkatesh/Good/venkatesh_good_19/{img_name}"])
