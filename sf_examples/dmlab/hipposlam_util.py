import os
import os.path
import pickle
import pandas as pd
import numpy as np

def midedges(edges):
    return (edges[:-1] + edges[1:]) / 2

def save_pickle(save_path, data):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

def read_pickle(read_path):
    with open(read_path, 'rb') as f:
        data = pickle.load(f)
    return data

class Recorder:
    def __init__(self, *args):
        self.records_dict = {key:[] for key in args}
        self.records_df = None

    def record(self, **kwargs):
        for key, val in kwargs.items():
            self.records_dict[key].append(val)

    def return_avers(self):
        return {key:np.mean(val) for key, val in self.records_dict.items()}

    def to_csv(self, save_pth):
        self.records_df = pd.DataFrame(self.records_dict)
        self.records_df.to_csv(save_pth, mode='a', header=not os.path.exists(save_pth))
    def clear_records_dict(self):
        for key in self.records_dict.keys():
            self.records_dict[key] = []

    def append_to_pickle(self, data_pth):
        if os.path.exists(data_pth):
            data = read_pickle(data_pth)  # list of records_dict
            data.append(self.records_dict)
            save_pickle(data_pth, data)
        else:
            self.save_as_pickle_for_append(data_pth)

    def save_as_pickle_for_append(self, data_pth):
        save_pickle(data_pth, [self.records_dict])


    def __getitem__(self, key):
        return self.records_dict[key]