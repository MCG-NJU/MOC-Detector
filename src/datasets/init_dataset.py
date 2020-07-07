from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.sampler import Sampler

from .dataset.ucf101 import UCF101
from .dataset.hmdb import JHMDB


switch_dataset = {
    'ucf101': UCF101,
    'hmdb': JHMDB,
}


def get_dataset(dataset):
    class Dataset(switch_dataset[dataset], Sampler):
        pass
    return Dataset
