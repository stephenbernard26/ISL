import pickle
from collections import Counter
if __name__ == '__main__':

    data1 = '/4TBHD/ISL/CodeBase/seq_dataset.pkl'
    data2 = '/4TBHD/ISL/CodeBase/Sequence_Model/v2_40sign_all.pkl'

    with open(data1, 'rb') as f:
        x = pickle.load(f)

    print((x))

    # with open(data2, 'rb') as f:
    #     y = pickle.load(f)
    

    # print(len(x))
    # print(len(y))
    
    # z = x | y

    # print(len(z))

    # # Extract and count each key after splitting
    # key_count = Counter(key.split('_')[1] for key in z.keys())

    # # Print the count of each key
    # for key, count in key_count.items():
    #     print(f"{key}: {count}")

    # with open("v2_1_40_sign_all_comb_seq.pkl", 'wb') as f:
    #     pickle.dump(z, f)

