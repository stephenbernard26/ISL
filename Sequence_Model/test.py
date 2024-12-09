# import numpy as np
# import mediapipe as mp
# import os
# import cv2

# mp_holistic = mp.solutions.holistic  # used to extract full body keypoints
# features_old = np.zeros((52, 2))

# # function for keypoint detection
# def mediapipe_detection(frame, model):
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame.flags.writeable = False
#     results = model.process(frame)
#     frame.flags.writeable = True
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     return frame, results

# # function to extract keypoints
# def extract_keypoints(results):
#     face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 3))
#     pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
#     lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
#     rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
#     return True, face, pose, lh, rh

# # function to extract useful features from the landmarks
# def extract_features(face, pose, left_hand, right_hand, frame_width, frame_height):
#     features = []

#     pose_filter = [11, 12, 13, 14, 15, 16]
#     face_filter = [4, 13, 68, 93, 151, 175, 207, 298, 427, 454, 33, 133, 362, 263]
#     lh_filter = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20]
#     rh_filter = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20]

#     for indx, lm in enumerate(face):
#         if indx in face_filter:
#             x = lm[0]
#             y = lm[1]
#             features.append([x, y])
#     for indx, lm in enumerate(pose):
#         if indx in pose_filter:
#             x = lm[0]
#             y = lm[1]
#             features.append([x, y])
#     for indx, lm in enumerate(left_hand):
#         if indx in lh_filter:
#             x = lm[0]
#             y = lm[1]
#             features.append([x, y])
#     for indx, lm in enumerate(right_hand):
#         if indx in rh_filter:
#             x = lm[0]
#             y = lm[1]
#             features.append([x, y])

#     return np.array(features)

# # Recursive function to process videos in all subdirectories
# def process_directory(dataset_dir, save_directory, frame_output_folder):
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
#                         frame, results = mediapipe_detection(frame, holistic)
#                         frame_height = frame.shape[0]
#                         frame_width = frame.shape[1]

#                         status, face, pose, left_hand, right_hand = extract_keypoints(results)
#                         if status == False:
#                             features = features_old
#                         else:
#                             features = extract_features(face, pose, left_hand, right_hand, frame_width, frame_height)
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

# # Define input/output directories
# dataset_dir = "/4TBHD/ISL/data_preparation/v2_40sign_videos"
# # save_directory = "../v2_40sign_videos_keypoints_onlyface/keypoints"
# # frame_output_folder = "../v2_40sign_videos_keypoints_onlyface/onlyface"

# save_directory = "./test/keypoints"
# frame_output_folder = "./test/onlyface"

# # Ensure the base output directories exist
# if not os.path.exists(save_directory):
#     os.makedirs(save_directory)

# if not os.path.exists(frame_output_folder):
#     os.makedirs(frame_output_folder)

# # Process all videos in the directory structure
# process_directory(dataset_dir, save_directory, frame_output_folder)


import pickle
from utility import *
seq_dict_path = '/4TBHD/ISL/CodeBase/Sequence_Model/pickle_files/5sign_5person_7video__vector_sequence_pkl/5sign_2video_5person.pkl'

# Load the data
with open(seq_dict_path, 'rb') as file:
    data = pickle.load(file)

config = read_config('config.yaml')

construct_seq_dictionary(config,data,"Reduced",save_path = '/4TBHD/ISL/CodeBase/Sequence_Model/pickle_files/5sign_5person_7video__vector_sequence_pkl/5sign_2video_5person_seq_path.pkl')