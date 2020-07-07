import os
import sys
import cPickle as pickle

from PIL import Image

CAFFE_PATH = os.path.dirname(os.path.dirname(__file__))
ROOT_DATASET_PATH = "/path/to/datasets/"  # CHANGEME
# If you use the ACT-detector scripts to download the action localization datasets, for instance UCF-Sports:
# ROOT_DATASET_PATH = CAFFE_PATH + "/data/UCFSports/"


class TubeDataset(object):
    """
    Abstract class for handling dataset of tubes.

    Here we assume that a pkl file exists as a cache. The cache is a dictionary with the following keys:
        labels: list of labels
        train_videos: a list with nsplits elements, each one containing the list of training videos
        test_videos: idem for the test videos
        nframes: dictionary that gives the number of frames for each video
        resolution: dictionary that output a tuple (h,w) of the resolution for each video
        gttubes: dictionary that contains the gt tubes for each video.
                    Gttubes are dictionary that associates from each index of label, a list of tubes.
                    A tube is a numpy array with nframes rows and 5 columns, <frame number> <x1> <y1> <x2> <y2>.
    """

    def __init__(self, dname, split=1):
        assert dname in ['UCFSports', 'JHMDB', 'UCF101'], "Unknown dataset name"
        self.NAME = dname
        self.SPLIT = split

        cache_file = os.path.join(CAFFE_PATH, 'cache', dname + '-GT.pkl')
        assert os.path.isfile(cache_file), "Missing cache file for dataset " + dname

        with open(cache_file, 'rb') as fid:
            cache = pickle.load(fid)

        for k in cache:
            setattr(self, ('_' if k != 'labels' else '') + k, cache[k])

    @property
    def nlabels(self):
        return len(self.labels)

    def train_vlist(self):
        return self._train_videos[self.SPLIT - 1]

    def test_vlist(self):
        return self._test_videos[self.SPLIT - 1]

    def nframes(self, v):
        return self._nframes[v]

    def resolution(self, v):
        return self._resolution[v]

    def gttubes(self, v):
        return self._gttubes[v]

    def imfile(self, v, i):
        raise NotImplementedError("TubeDataset is an abstract class, method imfile not implemented")

    def flowfile(self, v, i):
        raise NotImplementedError("TubeDataset is an abstract class, method flowfile not implemented")

    def frame_format(self, v, i):
        raise NotImplementedError("TubeDataset is an abstract class, method frame_format not implemented")


class UCFSports(TubeDataset):

    def __init__(self, split=1):
        assert split == 1, "UCFSports has only 1 split"
        super(UCFSports, self).__init__('UCFSports', 1)

    def imfile(self, v, i):
        return os.path.join(ROOT_DATASET_PATH, "UCFSports", "Frames", v, "{:0>6}.jpg".format(i))

    def flowfile(self, v, i):
        return os.path.join(ROOT_DATASET_PATH, "UCFSports", "FlowBrox04", v, "{:0>6}.jpg".format(i))

    def frame_format(self, v, i):
        return os.path.join(v, "{:0>6}".format(i))


class JHMDB(TubeDataset):
    def __init__(self, split=1):
        assert 1 <= split <= 3, "JHMDB have 3 splits"
        super(JHMDB, self).__init__('JHMDB', split)

    def imfile(self, v, i):
        return os.path.join(ROOT_DATASET_PATH, "JHMDB", "Frames", v, "{:0>5}.png".format(i))

    def flowfile(self, v, i):
        return os.path.join(ROOT_DATASET_PATH, "JHMDB", "FlowBrox04", v, "{:0>5}.jpg".format(i))

    def frame_format(self, v, i):
        return os.path.join(v, "{:0>5}".format(i))


class UCF101(TubeDataset):
    def __init__(self, split=1):
        assert split == 1, "We use only the first split of UCF101"
        super(UCF101, self).__init__('UCF101', 1)

    def imfile(self, v, i):
        return os.path.join(ROOT_DATASET_PATH, "UCF101", "Frames", v, "{:0>5}.jpg".format(i))

    def flowfile(self, v, i):
        return os.path.join(ROOT_DATASET_PATH, "UCF101", "FlowBrox04", v, "{:0>5}.jpg".format(i))

    def frame_format(self, v, i):
        return os.path.join(v, "{:0>5}".format(i))


def GetDataset(dname):
    assert dname in ['UCFSports', 'JHMDB', 'JHMDB2', 'JHMDB3', 'UCF101'], "Unknown dataset " + dname

    if dname == 'UCFSports':
        return UCFSports()
    if dname == 'JHMDB':
        return JHMDB()
    if dname == 'JHMDB2':
        return JHMDB(2)
    if dname == 'JHMDB3':
        return JHMDB(3)
    if dname == 'UCF101':
        return UCF101()
