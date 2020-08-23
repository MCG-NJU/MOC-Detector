from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import numpy as np
import torch
import time

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
        self.resolution = dataset._resolution
        self.input_h = dataset._resize_height
        self.input_w = dataset._resize_width
        self.output_h = self.input_h // self.opt.down_ratio
        self.output_w = self.input_w // self.opt.down_ratio
        self.indices = []
        for v in self.vlist:
            for i in range(1, 2000):
                # if not os.path.exists(self.outfile(v, i)):
                self.indices += [(v, i)]
        self.img_buffer = []
        self.img_buffer_flip = []
        self.last_video = -1
        self.last_frame = -1
        self.fake_image = np.random.rand(240, 320, 3).astype(np.float32)

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
                images = [self.fake_image for i in range(self.opt.K)]
                images = self.pre_process(images)
                if self.opt.flip_test:
                    self.img_buffer = images[:self.opt.K]
                    self.img_buffer_flip = images[self.opt.K:]
                else:
                    self.img_buffer = images

        else:
            if self.opt.rgb_model != '':
                image = self.fake_image
                image, image_flip = self.pre_process_single_frame(image)
                del self.img_buffer[0]
                self.img_buffer.append(image)
                if self.opt.flip_test:
                    del self.img_buffer_flip[0]
                    self.img_buffer_flip.append(image_flip)
                    images = self.img_buffer + self.img_buffer_flip
                else:
                    images = self.img_buffer

        return {'images': images, 'flows': flows, 'meta': {'height': h, 'width': w, 'output_height': self.output_h, 'output_width': self.output_w}, 'video_tag': video_tag}

    def __len__(self):
        return len(self.indices)


def speed_test_stream_inference(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    torch.backends.cudnn.benchmark = True
    if opt.flow_model != '':
        print('Online speed test does not support flow model.')
        sys.exit()

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

    for iter, data in enumerate(data_loader):

        if iter == 200:
            tag1 = iter
            time1 = time.time()

        if iter == 1200:
            tag2 = iter
            time2 = time.time()

            time_cost = time2 - time1
            total_frame = tag2 - tag1

            speed = time_cost / total_frame
            fps = total_frame / time_cost

            print('speed is: ', speed)
            print('fps is: ', fps)
            assert 0

        detections = detector.run(data)
