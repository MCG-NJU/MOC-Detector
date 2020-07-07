from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from progress.bar import Bar
import torch
import pickle
from multiprocessing import Process
import h5py

from opts import opts
from datasets.init_dataset import switch_dataset
from detector.speed_moc_det import MOCDetector
import random

GLOBAL_SEED = 317


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def worker_init_fn(dump):
    set_seed(GLOBAL_SEED)


class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.pre_process_func = pre_process_func
        self.opt = opt
        self.vlist = dataset._test_videos[dataset.split - 1]
        self.nframes = dataset._nframes
        self.imagefile = dataset.imagefile
        self.flowfile = dataset.flowfile
        self.indices = []
        for v in self.vlist:
            for i in range(1, 1 + self.nframes[v]):
                if not os.path.exists(self.outfile(v, i)):
                    self.indices += [(v, i)]

    def __getitem__(self, index):
        v, frame = self.indices[index]
        images = []
        flows = []

        if self.opt.rgb_model != '':
            images = [cv2.imread(self.imagefile(v, frame)).astype(np.float32)]
            images = self.pre_process_func(images)

        if self.opt.flow_model != '':
            flows = [cv2.imread(self.flowfile(v, min(frame + i, self.nframes[v]))).astype(np.float32) for i in range(self.opt.ninput)]
            flows = self.pre_process_func(flows, is_flow=True, ninput=self.opt.ninput)

        outfile = self.outfile(v, frame)
        if not os.path.isdir(os.path.dirname(outfile)):
            os.system("mkdir -p '" + os.path.dirname(outfile) + "'")

        return {'outfile': outfile, 'images': images, 'flows': flows}

    def outfile(self, v, i):
        return os.path.join(self.opt.inference_dir, 'inference_feature', v, "{:0>5}.feature".format(i))

    def __len__(self):
        return len(self.indices)


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.opt = opt
        self.vlist = dataset._test_videos[dataset.split - 1]
        self.nframes = dataset._nframes

        self.resolution = dataset._resolution
        self.input_h = dataset._resize_height
        self.input_w = dataset._resize_width
        self.output_h = self.input_h // self.opt.down_ratio
        self.output_w = self.input_w // self.opt.down_ratio
        self.indices = []
        for v in self.vlist:
            for i in range(1, 1 + self.nframes[v] - self.opt.K + 1):
                if not os.path.exists(self.outfile(v, i)):
                    self.indices += [(v, i)]

    def __getitem__(self, index):
        v, frame = self.indices[index]
        h, w = self.resolution[v]

        outfile = self.outfile(v, frame)
        if not os.path.isdir(os.path.dirname(outfile)):
            os.system("mkdir -p '" + os.path.dirname(outfile) + "'")
        rgb_features, rgb_features_flip, flow_features, flow_features_flip = [], [], [], []
        for i in range(self.opt.K):
            # feature = torch.load(self.feature_file(v, frame + i), map_location=lambda storage, loc: storage)
            # if self.opt.rgb_model != '':
            #     rgb_features.append(feature['rgb_features'])
            #     if self.opt.flip_test:
            #         rgb_features_flip.append(feature['rgb_features_flip'])
            # if self.opt.flow_model != '':
            #     flow_features.append(feature['flow_features'])
            #     if self.opt.flip_test:
            #         flow_features_flip.append(feature['flow_features_flip'])
            with h5py.File(self.feature_file(v, frame + i), 'r') as f:
                if self.opt.rgb_model != '':
                    rgb_features.append(f.get('rgb_features')[:])
                    if self.opt.flip_test:
                        rgb_features_flip.append(f.get('rgb_features_flip')[:])
                if self.opt.flow_model != '':
                    flow_features.append(f.get('flow_features')[:])
                    if self.opt.flip_test:
                        flow_features_flip.append(f.get('flow_features_flip')[:])
        return {'outfile': outfile, 'rgb_features': rgb_features, 'rgb_features_flip': rgb_features_flip, 'flow_features': flow_features, 'flow_features_flip': flow_features_flip, 'meta': {'height': h, 'width': w, 'output_height': self.output_h, 'output_width': self.output_w}}

    def outfile(self, v, i):
        return os.path.join(self.opt.inference_dir, v, "{:0>5}.pkl".format(i))

    def feature_file(self, v, i):
        return os.path.join(self.opt.inference_dir, 'inference_feature', v, "{:0>5}.feature".format(i))

    def __len__(self):
        return len(self.indices)


def speed_inference_backbone(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    torch.backends.cudnn.benchmark = True

    Dataset = switch_dataset[opt.dataset]
    opt = opts().update_dataset(opt, Dataset)

    dataset = Dataset(opt, 'test')
    detector = MOCDetector(opt)
    detector.load_backbone()
    prefetch_dataset = PrefetchDataset(opt, dataset, detector.pre_process)
    data_loader = torch.utils.data.DataLoader(
        prefetch_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        drop_last=False,
        worker_init_fn=worker_init_fn)

    num_iters = len(data_loader)

    bar = Bar(opt.exp_id, max=num_iters)

    print('inference chunk_sizes:', opt.chunk_sizes)
    for iter, data in enumerate(data_loader):
        outfile = data['outfile']

        rgb_features, rgb_features_flip, flow_features, flow_features_flip = detector.extract_feature(data)
        if opt.rgb_model != '':
            rgb_features = rgb_features.detach().cpu().numpy()
            if opt.flip_test:
                rgb_features_flip = rgb_features_flip.detach().cpu().numpy()
        if opt.flow_model != '':
            flow_features = flow_features.detach().cpu().numpy()
            if opt.flip_test:
                flow_features_flip = flow_features_flip.detach().cpu().numpy()
        # print('before save')
        p = Process(target=save_feature, args=(opt, outfile, rgb_features, rgb_features_flip, flow_features, flow_features_flip))
        p.start()

        Bar.suffix = 'inference_backbone: [{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            iter, num_iters, total=bar.elapsed_td, eta=bar.eta_td)

        bar.next()
    bar.finish()


def save_feature(opt, outfile, rgb_features, rgb_features_flip, flow_features, flow_features_flip):
    for i in range(len(outfile)):
        with h5py.File(outfile[i], 'w') as f:
            if opt.rgb_model != '':
                f.create_dataset("rgb_features", data=rgb_features[i], compression="lzf")
                if opt.flip_test:
                    f.create_dataset("rgb_features_flip", data=rgb_features_flip[i], compression="lzf")
            if opt.flow_model != '':
                f.create_dataset("flow_features", data=flow_features[i], compression="lzf")
                if opt.flip_test:
                    f.create_dataset("flow_features_flip", data=flow_features_flip[i], compression="lzf")


# def save_feature(outfile, rgb_features, rgb_features_flip, flow_features, flow_features_flip):
#     for i in range(len(outfile)):
#         feature = {}
#         if opt.rgb_model != '':
#             feature['rgb_features'] = rgb_features[i]
#             if opt.flip_test:
#                 feature['rgb_features_flip'] = rgb_features_flip[i]
#         if opt.flow_model != '':
#             feature['flow_features'] = flow_features[i]
#             if opt.flip_test:
#                 feature['flow_features_flip'] = flow_features_flip[i]
#         torch.save(feature, outfile[i])
#         # with open(outfile[i], 'wb') as file:
#         #     pickle.dump(feature, file)


def speed_inference_det(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    torch.backends.cudnn.benchmark = True

    Dataset = switch_dataset[opt.dataset]
    opt = opts().update_dataset(opt, Dataset)

    dataset = Dataset(opt, 'test')
    detector = MOCDetector(opt)
    detector.load_branch()
    feature_dataset = FeatureDataset(opt, dataset, detector.pre_process)
    data_loader = torch.utils.data.DataLoader(
        feature_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        drop_last=False,
        worker_init_fn=worker_init_fn)

    num_iters = len(data_loader)

    bar = Bar(opt.exp_id, max=num_iters)

    print('inference chunk_sizes:', opt.chunk_sizes)
    for iter, data in enumerate(data_loader):
        # if iter > 100:
        #     break
        outfile = data['outfile']

        detections = detector.run(data)

        # p = Process(target=save_detections, args=(outfile, detections))
        # p.start()
        for i in range(len(outfile)):
            with open(outfile[i], 'wb') as file:
                pickle.dump(detections[i], file)

        Bar.suffix = 'inference_detect: [{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            iter, num_iters, total=bar.elapsed_td, eta=bar.eta_td)

        bar.next()
    bar.finish()


# def save_detections(outfile, detections):
#     for i in range(len(outfile)):
#         with open(outfile[i], 'wb') as file:
#             pickle.dump(detections[i], file)
