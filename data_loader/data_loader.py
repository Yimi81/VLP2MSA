from base import BaseDataLoaderExplicitSplit, BaseMultiDataLoader

from data_loader.MOSI_dataset import MOSI
from data_loader.MOSEI_dataset import MOSEI
from data_loader.SIMS_dataset import SIMS
from data_loader.SIMS_V2_dataset import SIMS_V2

from data_loader.transforms import init_transform_dict


def dataset_loader(dataset_name,
                   text_params,
                   video_params,
                   data_dir,
                   metadata_dir=None,
                   split='train',
                   tsfms=None,
                   cut=None,
                   subsample=1,
                   sliding_window_stride=-1,
                   reader='decord'):
    kwargs = dict(
        dataset_name=dataset_name,
        text_params=text_params,
        video_params=video_params,
        data_dir=data_dir,
        metadata_dir=metadata_dir,
        split=split,
        tsfms=tsfms,
        cut=cut,
        subsample=subsample,
        sliding_window_stride=sliding_window_stride,
        reader=reader,
    )

    # TODO: change to...
    #  dataset = globals()[dataset_name]
    #  ...is this safe / or just lazy?
    if dataset_name == "MOSI":
        dataset = MOSI(**kwargs)
    elif dataset_name == "MOSEI":
        dataset = MOSEI(**kwargs)
    elif dataset_name == "SIMS":
        dataset = SIMS(**kwargs)
    elif dataset_name == "SIMS_V2":
        dataset = SIMS_V2(**kwargs)
    else:
        raise NotImplementedError(f"Dataset: {dataset_name} not found.")

    return dataset


class TextVideoDataLoader(BaseDataLoaderExplicitSplit):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 metadata_dir=None,
                 split='train',
                 tsfm_params=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='decord',
                 batch_size=1,
                 num_workers=1,
                 prefetch_factor=2,
                 shuffle=True,
                 val_batch_size=None):
        if tsfm_params is None:
            tsfm_params = {}
        tsfm_dict = init_transform_dict(**tsfm_params)

        tsfm = tsfm_dict[split]
        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, metadata_dir, split, tsfm, cut,
                                 subsample, sliding_window_stride, reader)
        if split != 'train':
            shuffle = False

        if val_batch_size is not None and split == 'val':
            batch_size = val_batch_size

        super().__init__(dataset, batch_size, shuffle, num_workers, prefetch_factor=prefetch_factor)
        self.dataset_name = dataset_name


class TextVideoMultiDataLoader(BaseMultiDataLoader):
    # TODO: figure out neat way to have N data_loaders
    # TODO: also add N weighted sampler
    def __init__(self, data_loader1, data_loader2):
        # get class from "type" in dict
        dls_cfg = [data_loader1, data_loader2]
        dls = []
        for dcfg in dls_cfg:
            dl = globals()[dcfg['type']](**dcfg['args'])
            dls.append(dl)
        super().__init__(dls)


if __name__ == "__main__":
    kwargs = {
        "dataset_name": "CondensedMovies",
        "data_dir": "/scratch/shared/beegfs/maxbain/datasets/CondensedMovies",
        "shuffle": True,
        "num_workers": 16,
        "batch_size": 16,
        "split": "train",
        "cut": "chall",
        "subsample": 1,
        "text_params": {
            "input": "text",
            "caption_replace_prob": 0.5
        },
        "video_params": {
            "extraction_fps": 25,
            "extraction_res": 256,
            "input_res": 224,
            "num_frames": 4,
            "shot_replace": True,
            "shot_replace_prob": 0.5
        },
        "tsfm_params": {"auto_aug": True}
       }

    dl = TextVideoDataLoader(**kwargs)
    dl.dataset.__getitem__(0)
    for x in range(1000):
        res = next(iter(dl))
        print(x)
