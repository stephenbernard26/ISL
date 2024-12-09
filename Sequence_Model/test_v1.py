import pickle
from utility_v1 import *
seq_dict_path = '/4TBHD/ISL/CodeBase/Sequence_Model/ai4b_5sign.pkl'

# Load the data
with open(seq_dict_path, 'rb') as file:
    data = pickle.load(file)

config = read_config('config.yaml')

construct_seq_dictionary(config,data,"Reduced",save_path = '/4TBHD/ISL/CodeBase/Sequence_Model/pickle_files/ai4b/ai4b_5sign_seq_vector.pkl')