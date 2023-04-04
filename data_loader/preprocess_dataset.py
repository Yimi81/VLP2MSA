from ast import Import
import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
import pandas as pd
import torch
from subprocess import check_call, CalledProcessError
from collections import defaultdict
from tqdm import tqdm_notebook

############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################

class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='mosei', split_type='train', if_align=False):
        super(Multimodal_Datasets, self).__init__()
        new_dataset_path = os.path.join(dataset_path, data, "Processed", 'aligned_50.pkl' if if_align else 'unaligned_50.pkl')
        dataset = pickle.load(open(new_dataset_path, 'rb'))

        # Create torch tensors
        self.text = dataset[split_type]['text'].astype(np.float32)
        self.raw_text = dataset[split_type]['raw_text']

        self.vision = dataset[split_type]['vision'].astype(np.float32)
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.labels = dataset[split_type]['regression_labels'].astype(np.float32)

        self.n_modalities = 3 # vision/ text/ audio

    def get_n_modalities(self):
        return self.n_modalities
    
    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]
    
    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[0], self.labels.shape[1]

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        return self.text[index], self.raw_text[index], self.audio[index], self.vision[index], self.labels[index]  
