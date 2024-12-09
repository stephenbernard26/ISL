"""
Author: Sandeep Kumar Suresh
        EE23S059
"""


from torch.utils.data import Dataset
import torch

class SequenceClassificationDataset(Dataset):

    def __init__(self, features,labels): #,dataset)



        self.modified_feat , self.modified_label = [],[]

        # self.features = dataset["feature_list"].tolist()      
        # self.labels = dataset["labels"].tolist() 
        self.features = features      
        self.labels =  labels

        # self.label_mapping = {
        #     'go': 0,
        #     'cook': 1,
        #     'thankyou': 2,
        #     'beautiful': 3,
        #     'good': 4,
        #     'dead': 5,
        #     'sorry': 6,
        #     'hello': 7,
        #     'brush': 8,
        #     'bye': 9,
        #     'summer': 10,
        #     'cup': 11,
        #     'door': 12,
        #     'room': 13,
        #     'bed': 14,
        #     'black': 15,
        #     'winter': 16,
        #     'fish': 17,
        #     'aeroplane': 18,
        #     'teacher': 19,
        #     'goodmorning': 20,
        #     'dog': 21,
        #     'mirror': 22,
        #     'doctor': 23,
        #     'train': 24,
        #     'above': 25,
        #     'shoot': 26,
        #     'bowl': 27,
        #     'cat': 28,
        #     'coffee': 29,

        # }

        # self.label_mapping = {
        #     'dustbin': 0,
        #     'earth': 1,

        #     'below': 2,
        #     'garden': 3,

        #     'coat': 4,
        
        #     }

        # self.label_mapping = {
        # 'beautiful': 0,
        # 'brush': 1,
        # 'bye': 2,
        # 'cook': 3,
        # 'dead': 4,
        # 'go': 5,
        # 'good': 6,
        # 'hello': 7,
        # 'sorry': 8,
        # 'thankyou': 9,
        # 'above': 10,
        # 'aeroplane': 11,
        # 'bed': 12,
        # 'below': 13,
        # 'black': 14,
        # 'bob': 15,
        # 'bowl': 16,
        # 'cat': 17,
        # 'coat': 18,
        # 'coffee': 19,
        # 'conductor': 20,
        # 'cup': 21,
        # 'doctor': 22,
        # 'dog': 23,
        # 'door': 24,
        # 'dustbin': 25,
        # 'earmuffs': 26,
        # 'earth': 27,
        # 'family': 28,
        # 'fish': 29,
        # 'garden': 30,
        # 'gift': 31,
        # 'glasses': 32,
        # 'goodmorning': 33,
        # 'hairband': 34,
        # 'key': 35,
        # 'lock': 36,
        # 'lorry': 37,
        # 'mirror': 38,
        # 'party': 39,
        # 'room': 40,
        # 'shoot': 41,
        # 'summer': 42,
        # 'swimming': 43,
        # 'talk': 44,
        # 'teacher': 45,
        # 'thursday': 46,
        # 'train': 47,
        # 'tubelight': 48,
        # 'winter': 49
        # }

        # self.label_mapping = {
        #             'dustbin': 0,
        #             'family': 1,
        #             'tubelight': 2,
        #             'earth': 3,
        #             'hairband': 4,
        #             'earmuffs': 5,
        #             'below': 6,
        #             'garden': 7,
        #             'gift': 8,
        #             'lock': 9,
        #             'talk': 10,
        #             'key': 11,
        #             'conductor': 12,
        #             'glasses': 13,
        #             'lorry': 14,
        #             'thursday': 15,
        #             'bob': 16,
        #             'party': 17,
        #             'coat': 18,
        #             'swimming': 19
        #         }


        # self.label_mapping = {
        #             'dustbin': 0,   
        #             'coat': 1,

        #             'swimming': 2,
        #             'key': 3,

        #             'lock': 4
        #         }

        # self.label_mapping = {
        #             'adult': 0,   
        #             'blue': 1,

        #             'baby': 2,
        #             'bicycle': 3,

        #             'i': 4
        #         }

        # self.label_mapping = {
        #     "adult": 0,
        #     "alright": 1,
        #     "animal": 2,
        #     "baby": 3,
        #     "bad": 4,
        #     "bank": 5,
        #     "bicycle": 6,
        #     "biglarge": 7,
        #     "bird": 8,
        #     "black": 9,
        #     "blue": 10,
        #     "boat": 11,
        #     "boy": 12,
        #     "brother": 13,
        #     "brown": 14,
        #     "bus": 15,
        #     "car": 16,
        #     "cat": 17,
        #     "child": 18,
        #     "city": 19,
        #     "clothing": 20,
        #     "cold": 21,
        #     "colour": 22,
        #     "cool": 23,
        #     "court": 24,
        #     "cow": 25,
        #     "crowd": 26,
        #     "daughter": 27,
        #     "dog": 28,
        #     "dress": 29,
        #     "dry": 30,
        #     "fast": 31,
        #     "father": 32,
        #     "fish": 33,
        #     "friend": 34,
        #     "girl": 35,
        #     "good": 36,
        #     "goodafternoon": 37,
        #     "goodevening": 38,
        #     "goodmorning": 39,
        #     "goodnight": 40,
        #     "green": 41,
        #     "grey": 42,
        #     "ground": 43,
        #     "happy": 44,
        #     "hat": 45,
        #     "he": 46,
        #     "healthy": 47,
        #     "hello": 48,
        #     "horse": 49,
        #     "hospital": 50,
        #     "hot": 51,
        #     "house": 52,
        #     "howareyou": 53,
        #     "husband": 54,
        #     "i": 55,
        #     "india": 56,
        #     "it": 57,
        #     "king": 58,
        #     "library": 59,
        #     "location": 60,
        #     "long": 61,
        #     "loud": 62,
        #     "man": 63,
        #     "market": 64,
        #     "mother": 65,
        #     "mouse": 66,
        #     "narrow": 67,
        #     "neighbour": 68,
        #     "new": 69,
        #     "office": 70,
        #     "old": 71,
        #     "orange": 72,
        #     "pant": 73,
        #     "parent": 74,
        #     "park": 75,
        #     "pink": 76,
        #     "plane": 77,
        #     "player": 78,
        #     "pleased": 79,
        #     "pocket": 80,
        #     "president": 81,
        #     "queen": 82,
        #     "quiet": 83,
        #     "red": 84,
        #     "restaurant": 85,
        #     "school": 86,
        #     "she": 87,
        #     "shirt": 88,
        #     "shoes": 89,
        #     "short": 90,
        #     "sick": 91,
        #     "sister": 92,
        #     "skirt": 93,
        #     "slow": 94,
        #     "smalllittle": 95,
        #     "son": 96,
        #     "storeorshop": 97,
        #     "streetorroad": 98,
        #     "suit": 99,
        #     "tall": 100,
        #     "temple": 101,
        #     "thankyou": 102,
        #     "they": 103,
        #     "train": 104,
        #     "trainstation": 105,
        #     "trainticket": 106,
        #     "transportation": 107,
        #     "truck": 108,
        #     "t-shirt": 109,
        #     "university": 110,
        #     "warm": 111,
        #     "we": 112,
        #     "wet": 113,
        #     "white": 114,
        #     "wide": 115,
        #     "wife": 116,
        #     "woman": 117,
        #     "yellow": 118,
        #     "you": 119,
        #     "young": 120,
        #     "you(plural)": 121
        # }

        self.label_mapping = {
            "bank": 0,
            "biglarge": 1,
            "bird": 2,
            "black": 3,
            "boy": 4,
            "brother": 5,
            "car": 6,
            "cellphone": 7,
            "court": 8,
            "cow": 9,
            "death": 10,
            "dog": 11,
            "dry": 12,
            "election": 13,
            "fall": 14,
            "fan": 15,
            "father": 16,
            "girl": 17,
            "good": 18,
            "goodmorning": 19,
            "happy": 20,
            "hat": 21,
            "hello": 22,
            "hot": 23,
            "house": 24,
            "i": 25,
            "it": 26,
            "long": 27,
            "loud": 28,
            "monday": 29,
            "new": 30,
            "paint": 31,
            "pen": 32,
            "priest": 33,
            "quiet": 34,
            "red": 35,
            "shoes": 36,
            "short": 37,
            "smalllittle": 38,
            "storeorshop": 39,
            "summer": 40,
            "teacher": 41,
            "thankyou": 42,
            "time": 43,
            "trainticket": 44,
            "t-shirt": 45,
            "white": 46,
            "window": 47,
            "year": 48,
            "you(plural)": 49
        }




        for feat,label in zip(self.features,self.labels):
            # print(feat)
            self.modified_feat.append(torch.tensor(feat,dtype = torch.float))
            self.modified_label.append(self.label_mapping.get(label))



 
    def __len__(self):
        """
        Note: We are getting the length of the list
        """
        return len(self.modified_label)

    
    def __getitem__(self, idx):
        
        return self.modified_feat[idx],self.modified_label[idx]

