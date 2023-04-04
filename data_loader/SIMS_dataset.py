import json
import os
import random

import numpy as np
import pandas as pd

from base.base_dataset import TextVideoDataset


class SIMS(TextVideoDataset):
    def _load_metadata(self):
        assert self.split in ["train", "val", "test"]
        split = self.split
        if split == "val":
            split = "valid"

        csv_fp = os.path.join(self.metadata_dir, 'label.csv')

        df = pd.read_csv(csv_fp)
        
        df = df[df['mode'] == split]

        d_dict = df.to_dict('list')

        self.metadata = {}
        for i in range(len(d_dict['video_id'])):
            video_id = d_dict['video_id'][i] + '/' + str(d_dict['clip_id'][i]).zfill(4)
            self.metadata[len(self.metadata)] = (video_id, d_dict['text'][i], d_dict['label'][i])
       
        self.metadata = pd.DataFrame({'data': self.metadata})


    def _get_video_path(self, sample):

        video_dir_file = sample['data'][0]
        video, clip = video_dir_file.split("/")

        return os.path.join(self.data_dir, 'Raw', video, clip + '.mp4'), clip + '.mp4'

    def _get_caption(self, sample):
        
        return sample['data'][1]

    def _get_label(self, sample):

        return  sample['data'][2]