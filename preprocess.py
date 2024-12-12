"""
Author : Sandeep Kumar Suresh

Code for preprocessing the video data into npy files and frames.

The code process video frames and converts it into npy files based on the filter
config and also save the corresponding frames in the desired directory

    TODO:
        1. Code to bypass frame saving in order to save memory
        2. Provide a Config file for easy and fast experimentations
"""

import numpy as np
import mediapipe as mp
import os
import cv2
from dataclasses import dataclass
from typing import List
import yaml 
from concurrent.futures import ThreadPoolExecutor
import time

@dataclass
class Feature_Filters():

    face_filter: List[int] 
    pose_filter: List[int] 
    lh_filter: List[int] 
    rh_filter: List[int] 


@dataclass
class Feature_Extraction():

    filters: Feature_Filters

    # function for keypoint detection
    def mediapipe_detection(self,frame, model):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = model.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame, results

    # function to extract keypoints
    def extract_keypoints(self,results):
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 3))
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
        return True, face, pose, lh, rh

    # function to extract useful features from the landmarks
    def extract_features(self,face, pose, left_hand, right_hand, frame_width, frame_height):
        features = []

        for indx, lm in enumerate(face):
            if indx in self.filters.face_filter:
                x = lm[0]
                y = lm[1]
                features.append([x, y])
        for indx, lm in enumerate(pose):
            if indx in self.filters.pose_filter:
                x = lm[0]
                y = lm[1]
                features.append([x, y])
        for indx, lm in enumerate(left_hand):
            if indx in self.filters.lh_filter:
                x = lm[0]
                y = lm[1]
                features.append([x, y])
        for indx, lm in enumerate(right_hand):
            if indx in self.filters.rh_filter:
                x = lm[0]
                y = lm[1]
                features.append([x, y])

        return np.array(features)


    # # Recursive function to process videos in all subdirectories
    # def process_directory(self,dataset_dir, save_directory, frame_output_folder):

    #     mp_holistic = mp.solutions.holistic  # used to extract full body keypoints
    #     features_old = np.zeros((52, 2))

    #     for root, dirs, files in os.walk(dataset_dir):
    #         for video_file in files:
    #             video_extensions = ['.mp4', '.mov', '.MOV', '.MP4']
    #             if not any(video_file.endswith(ext) for ext in video_extensions):
    #                 continue

    #             # Construct paths
    #             relative_path = os.path.relpath(root, dataset_dir)
    #             video_path = os.path.join(root, video_file)

    #             save_video_path = os.path.join(save_directory, relative_path, video_file[:-4])
    #             os.makedirs(save_video_path, exist_ok=True)

    #             frame_output_path = os.path.join(frame_output_folder, relative_path, video_file[:-4])
    #             os.makedirs(frame_output_path, exist_ok=True)

    #             # Process the video
    #             with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    #                 cap = cv2.VideoCapture(video_path)
    #                 frame_count = 0
    #                 while cap.isOpened():
    #                     ret, frame = cap.read()
    #                     if ret:
    #                         frame_count += 1
    #                         frame, results = self.mediapipe_detection(frame, holistic)
    #                         frame_height = frame.shape[0]
    #                         frame_width = frame.shape[1]

    #                         status, face, pose, left_hand, right_hand = self.extract_keypoints(results)
    #                         if status == False:
    #                             features = features_old
    #                         else:
    #                             features = self.extract_features(face, pose, left_hand, right_hand, frame_width, frame_height)
    #                             features_old = features

    #                         # Save keypoints
    #                         npy_path = os.path.join(save_video_path, f"{frame_count}.npy")
    #                         np.save(npy_path, features)

    #                         # Draw keypoints on frame and save the frame
    #                         for lm in features:
    #                             x = int(lm[0] * frame_width)
    #                             y = int(lm[1] * frame_height)
    #                             cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

    #                         frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    #                         save_path_frame = os.path.join(frame_output_path, f"{frame_count}.jpg")
    #                         cv2.imwrite(save_path_frame, frame_resized)
    #                         # print(f"Key frame {frame_count} saved at: {save_path_frame}")
    #                     else:
    #                         break
    #                 cap.release()
    #                 print(save_path_frame)
    #                 cv2.destroyAllWindows()


    def process_dir_video(self,video_dir,frame_output_path):


        os.makedirs(frame_output_path,exist_ok=True)

        dir_name = ["black" , "fall","girl","good","goodmorning","happy","loud","pen","quiet","smalllittle","thank you","white","year","you(plural)"]

        for video_folder in os.listdir(video_dir):
            print(video_folder)
            if video_folder in dir_name:
                for video_name in os.listdir(os.path.join(video_dir,video_folder)):
                    video_path = os.path.join(video_dir,video_folder,video_name).strip()
                    print(video_path)


                    cap = cv2.VideoCapture(video_path.strip())
                    frame_count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        # print(ret)
                        if ret:
                            frame_count += 1
                            frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                            save_path_frame = os.path.join(frame_output_path, video_name, f"{frame_count}.jpg")
                            cv2.imwrite(save_path_frame, frame_resized)
                        else:
                            print(f"Error processing {video_path}")
                            break
                    cap.release()
                    # cv2.destroyAllWindows()




######################## Below Added Code to Test Multithreading ###########################

    def process_video(self, video_path, save_video_path ):#, frame_output_path):
        mp_holistic = mp.solutions.holistic

        # Process the video
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret: 
                    frame_count += 1
                    frame, results = self.mediapipe_detection(frame, holistic)
                    frame_height = frame.shape[0]
                    frame_width = frame.shape[1]

                    status, face, pose, left_hand, right_hand = self.extract_keypoints(results)
                    if status == False:
                        features = self.features_old
                    else:
                        features = self.extract_features(face, pose, left_hand, right_hand, frame_width, frame_height)
                        self.features_old = features

                    # Save keypoints
                    npy_path = os.path.join(save_video_path, f"{frame_count}.npy")
                    np.save(npy_path, features)

                    # # Draw keypoints on frame and save the frame
                    # for lm in features:
                    #     x = int(lm[0] * frame_width)
                    #     y = int(lm[1] * frame_height)
                    #     cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

                    # frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    # save_path_frame = os.path.join(frame_output_path, f"{frame_count}.jpg")
                    # cv2.imwrite(save_path_frame, frame_resized)
                else:
                    break
            cap.release()
            cv2.destroyAllWindows()


    def process_directory(self, dataset_dir, save_directory): #, frame_output_folder):
        
        dir_to_process = ['anirudh','rounit','stephen','varmija','vinayak','cindrella','ishan','sandeep','surendhar','venkatesh','ranadeep']
        video_tasks = []

        for root, dirs, files in os.walk(dataset_dir):

            # Get the top-level directory relative to the dataset_dir
            relative_path = os.path.relpath(root, dataset_dir)
            
            top_level_dir = relative_path.split(os.sep)[0]  # Extract the first folder name

            
            # Process only if the top-level directory is in dir_to_process
            if top_level_dir not in dir_to_process:
                continue

            for video_file in files:
                video_extensions = ['.mp4', '.mov', '.MOV', '.MP4']
                if not any(video_file.endswith(ext) for ext in video_extensions):
                    continue

                # Construct paths
                relative_path = os.path.relpath(root, dataset_dir)
                video_path = os.path.join(root, video_file)
            
                save_video_path = os.path.join(save_directory, relative_path, video_file[:-4])
                os.makedirs(save_video_path, exist_ok=True)

                # frame_output_path = os.path.join(frame_output_folder, relative_path, video_file[:-4])
                # os.makedirs(frame_output_path, exist_ok=True)

                # Add task for each video to be processed
                video_tasks.append((video_path, save_video_path ))#, frame_output_path))

        # # Process videos concurrently
        # with ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(self.process_video, video_path, save_video_path, frame_output_path)
        #                for video_path, save_video_path, frame_output_path in video_tasks]

        # Process videos concurrently
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_video, video_path, save_video_path ) #, frame_output_path)
                    for video_path, save_video_path  in video_tasks]

            # Wait for all threads to complete
            for future in futures:
                future.result()


def read_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == '__main__':

    # Reading the Config File
    config = read_config('config.yaml')

    # # Reading the directories from the Config Files
    # dataset_dir = config['paths']['dataset_dir']
    # save_directory = config['paths']['save_directory']
    # frame_output_folder = config['paths']['frame_output_folder']


    dataset_dir = '/4TBHD/ISL/SIGN_LANG_AUG'
    save_directory = '/4TBHD/ISL/data_preparation/test_all/54_keypoints'
    # frame_output_folder = '/4TBHD/ISL/data_preparation/v2_40sign_videos_54kpt_face/'


    # Reading the filter Configuration from the Config Files
    feature_filters = Feature_Filters(
        face_filter=config['filters']['face_filter'],
        pose_filter=config['filters']['pose_filter'],
        lh_filter=config['filters']['lh_filter'],
        rh_filter=config['filters']['rh_filter']
    )


    # Ensure the base output directories exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # if not os.path.exists(frame_output_folder):
    #     os.makedirs(frame_output_folder)

    feature_extractor = Feature_Extraction(filters=feature_filters)

    start = time.time()

    # # Process all videos in the directory structure
    # feature_extractor.process_directory(dataset_dir, save_directory, frame_output_folder)

    # Process all videos in the directory structure
    feature_extractor.process_directory(dataset_dir, save_directory)

    end = time.time()

    print("time needed for Execution : ", end - start)

    # frame_save_path = '/4TBHD/include_underperforming_class'
    # feature_extractor.process_dir_video('/4TBHD/include_50',frame_save_path)