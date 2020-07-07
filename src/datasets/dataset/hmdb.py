from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
from .base_dataset import BaseDataset


class JHMDB(BaseDataset):
    num_classes = 21

    def __init__(self, opt, mode):
        self.ROOT_DATASET_PATH = os.path.join(opt.root_dir, 'data/JHMDB')
        pkl_filename = 'JHMDB-GT.pkl'
        super(JHMDB, self).__init__(opt, mode, self.ROOT_DATASET_PATH, pkl_filename)

    def imagefile(self, v, i):
        return os.path.join(self.ROOT_DATASET_PATH, 'Frames', v, '{:0>5}.png'.format(i))

    def flowfile(self, v, i):
        return os.path.join(self.ROOT_DATASET_PATH, 'FlowBrox04', v, '{:0>5}.jpg'.format(i))
