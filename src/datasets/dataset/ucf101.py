from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from .base_dataset import BaseDataset


class UCF101(BaseDataset):
    num_classes = 24

    def __init__(self, opt, mode):
        assert opt.split == 1, "We use only the first split of UCF101"
        self.ROOT_DATASET_PATH = os.path.join(opt.root_dir, 'data/ucf24')
        pkl_filename = 'UCF101v2-GT.pkl'
        super(UCF101, self).__init__(opt, mode, self.ROOT_DATASET_PATH, pkl_filename)

    def imagefile(self, v, i):
        return os.path.join(self.ROOT_DATASET_PATH, 'rgb-images', v, '{:0>5}.jpg'.format(i))

    def flowfile(self, v, i):
        return os.path.join(self.ROOT_DATASET_PATH, 'brox-images', v, '{:0>5}.jpg'.format(i))
