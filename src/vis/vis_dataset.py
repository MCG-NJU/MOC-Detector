from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch.utils.data as data
import os


class VisualizationDataset(data.Dataset):

    def __init__(self, opt):

        super(VisualizationDataset, self).__init__()
        self.K = opt.K
        self.opt = opt
        self.inference_dir = opt.inference_dir

        self._ninput = opt.ninput
        self._resize_height = opt.resize_height
        self._resize_width = opt.resize_width
        self._nframes = 0
        for jpg in os.listdir(os.path.join(self.inference_dir, 'rgb')):
            if jpg.endswith('.jpg') or jpg.endswith('.png'):
                self._nframes += 1

    def imagefile(self, fid):
        return os.path.join(self.inference_dir, 'rgb', '{:0>5}.jpg'.format(fid))

    def flowfile(self, fid):
        return os.path.join(self.inference_dir, 'flow', '{:0>5}.jpg'.format(fid))
