from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from progress.bar import Bar
import torch
import pickle

from opts import opts
from datasets.init_dataset import switch_dataset
from detector.stream_moc_det import MOCDetector
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
    def __init__(self, opt, dataset, pre_process, pre_process_single_frame):
        self.pre_process = pre_process
        self.pre_process_single_frame = pre_process_single_frame
        self.opt = opt
        self.vlist = dataset._test_videos[dataset.split - 1]
        self.gttubes = dataset._gttubes
        self.nframes = dataset._nframes
        self.imagefile = dataset.imagefile
        self.flowfile = dataset.flowfile
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
        self.img_buffer = []
        self.flow_buffer = []
        self.img_buffer_flip = []
        self.flow_buffer_flip = []
        self.last_video = -1
        self.last_frame = -1

    def __getitem__(self, index):
        v, frame = self.indices[index]
        h, w = self.resolution[v]
        images = []
        flows = []
        video_tag = 0

        # if there is a new video
        if v == self.last_video and frame == self.last_frame + 1:
            video_tag = 1
        else:

            video_tag = 0

        self.last_video = v
        self.last_frame = frame

        if video_tag == 0:
            if self.opt.rgb_model != '':
                images = [cv2.imread(self.imagefile(v, frame + i)).astype(np.float32) for i in range(self.opt.K)]
                images = self.pre_process(images)
                if self.opt.flip_test:
                    self.img_buffer = images[:self.opt.K]
                    self.img_buffer_flip = images[self.opt.K:]
                else:
                    self.img_buffer = images

            if self.opt.flow_model != '':
                flows = [cv2.imread(self.flowfile(v, min(frame + i, self.nframes[v]))).astype(np.float32) for i in range(self.opt.K + self.opt.ninput - 1)]
                flows = self.pre_process(flows, is_flow=True, ninput=self.opt.ninput)

                if self.opt.flip_test:
                    self.flow_buffer = flows[:self.opt.K]
                    self.flow_buffer_flip = flows[self.opt.K:]
                else:
                    self.flow_buffer = flows

        else:
            if self.opt.rgb_model != '':
                image = cv2.imread(self.imagefile(v, frame + self.opt.K - 1)).astype(np.float32)
                image, image_flip = self.pre_process_single_frame(image)
                del self.img_buffer[0]
                self.img_buffer.append(image)
                if self.opt.flip_test:
                    del self.img_buffer_flip[0]
                    self.img_buffer_flip.append(image_flip)
                    images = self.img_buffer + self.img_buffer_flip
                else:
                    images = self.img_buffer

            if self.opt.flow_model != '':
                flow = cv2.imread(self.flowfile(v, min(frame + self.opt.K + self.opt.ninput - 2, self.nframes[v]))).astype(np.float32)
                data_last_flip = self.flow_buffer_flip[-1] if self.opt.flip_test else None
                data_last = self.flow_buffer[-1]
                flow, flow_flip = self.pre_process_single_frame(flow, is_flow=True, ninput=self.opt.ninput, data_last=data_last, data_last_flip=data_last_flip)
                del self.flow_buffer[0]
                self.flow_buffer.append(flow)
                if self.opt.flip_test:
                    del self.flow_buffer_flip[0]
                    self.flow_buffer_flip.append(flow_flip)
                    flows = self.flow_buffer + self.flow_buffer_flip
                else:
                    flows = self.flow_buffer

        outfile = self.outfile(v, frame)
        if not os.path.isdir(os.path.dirname(outfile)):
            os.system("mkdir -p '" + os.path.dirname(outfile) + "'")

        return {'outfile': outfile, 'images': images, 'flows': flows, 'meta': {'height': h, 'width': w, 'output_height': self.output_h, 'output_width': self.output_w}, 'video_tag': video_tag}

    def outfile(self, v, i):
        return os.path.join(self.opt.inference_dir, v, "{:0>5}.pkl".format(i))

    def __len__(self):
        return len(self.indices)


def stream_inference(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # torch.backends.cudnn.benchmark = True

    Dataset = switch_dataset[opt.dataset]
    opt = opts().update_dataset(opt, Dataset)

    dataset = Dataset(opt, 'test')
    detector = MOCDetector(opt)
    prefetch_dataset = PrefetchDataset(opt, dataset, detector.pre_process, detector.pre_process_single_frame)
    data_loader = torch.utils.data.DataLoader(
        prefetch_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=worker_init_fn)

    num_iters = len(data_loader)

    bar = Bar(opt.exp_id, max=num_iters)

    for iter, data in enumerate(data_loader):

        outfile = data['outfile']

        detections = detector.run(data)

        for i in range(len(outfile)):
            with open(outfile[i], 'wb') as file:
                pickle.dump(detections[i], file)

        Bar.suffix = 'inference: [{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            iter, num_iters, total=bar.elapsed_td, eta=bar.eta_td)

        bar.next()
    bar.finish()
