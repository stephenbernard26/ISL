import numpy as np
import yaml


if __name__ == '__main__':

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    npy_file = '/4TBHD/ISL/CodeBase/Sequence_Model/seq_npy_folder/janaghan_beautiful_1/12.npy'

    features = np.load(npy_file)

    pose_filter_config = config['filter_mapping']['pose_filter']

    desired_pose_filters = [11, 13, 15]

    selected_indices = [pose_filter_config[pf] for pf in desired_pose_filters]

    selected_features = features[selected_indices]

    print(selected_features)
