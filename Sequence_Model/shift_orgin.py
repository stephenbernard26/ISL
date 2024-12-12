import numpy as np
import os

def find_center_of_quadrilateral(points):
    # Calculate the center by averaging the x and y coordinates of the 4 points
    center_x = np.mean([points[14][0], points[21][0]])
    center_y = np.mean([points[14][1], points[21][1]])
    return center_x, center_y


def shift_keypoints_to_new_orgin(npy_path):
    """
    This function takes in the npy files and returns the file with shifted orgin.
    
    """

    # Load the .npy file
    # data = np.load(npy_path)
    data = npy_path
    missing_indices = [i for i in [14, 15, 20, 21] if data[i] is None]
    if missing_indices:
        print(f"{npy_path} is missing points at indices: {missing_indices}")
    
    center_x, center_y = find_center_of_quadrilateral(data)
    
    shifted_data = np.array([[point[0] - center_x, point[1] - center_y] for point in data])

    return shifted_data



def process_npy_files(directory_path, output_directory):
    # Ensure the root output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.npy'):
                npy_path = os.path.join(root, file)
                
                try:
                    # Load the .npy file
                    data = np.load(npy_path)
                    missing_indices = [i for i in [14, 15, 20, 21] if data[i] is None]
                    if missing_indices:
                        print(f"{file} is missing points at indices: {missing_indices}")
                        continue
                    
                    # Find the center of the quadrilateral
                    center_x, center_y = find_center_of_quadrilateral(data)
                    
                    # Subtract the center from each point
                    normalized_data = np.array([[point[0] - center_x, point[1] - center_y] for point in data])

                    # Create the corresponding subfolder structure in the output directory
                    # Get the relative path of the current file (relative to the root input directory)
                    relative_path = os.path.relpath(root, directory_path)
                    
                    # Create the corresponding subfolder in the output directory
                    output_subfolder = os.path.join(output_directory, relative_path)
                    os.makedirs(output_subfolder, exist_ok=True)
                    
                    # Save the normalized data to the new subfolder with the same file name
                    output_path = os.path.join(output_subfolder, file)
                    np.save(output_path, normalized_data)
                
                except IndexError as e:
                    print(f"IndexError in {file},{root}: Missing indices or wrong data format. Error: {e}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")



if __name__ == '__main__':
    # Example usage
    directory_path = '/4TBHD/ISL/data_preparation/test_blank_keypoints/Keypointsw' #Example path : /4TBHD/ISL/data_preparation/test_all/keypoints
    output_directory = '/4TBHD/ISL/data_preparation/test_all/shifted_origin__keypoints'#Example path : /4TBHD/ISL/data_preparation/normalised_keypoints
    process_npy_files(directory_path, output_directory)