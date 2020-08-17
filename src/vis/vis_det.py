from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import torch
import pickle
import sys

from tiny_opt import opts
from vis_dataset import VisualizationDataset
from build import build_tubes
from vis_utils import pkl_decode, vis_bbox, rgb2avi, video2frames, rgb2gif
sys.path.append("..")
from detector.stream_moc_det import MOCDetector


class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process, pre_process_single_frame):
        self.pre_process = pre_process
        self.pre_process_single_frame = pre_process_single_frame
        self.opt = opt
        self.nframes = dataset._nframes
        self.imagefile = dataset.imagefile
        self.flowfile = dataset.flowfile

        self.input_h = dataset._resize_height
        self.input_w = dataset._resize_width
        self.output_h = self.input_h // self.opt.down_ratio
        self.output_w = self.input_w // self.opt.down_ratio
        self.indices = []
        for i in range(1, 1 + self.nframes - self.opt.K + 1):
            if not os.path.exists(self.outfile(i)):
                self.indices.append(i)
        self.img_buffer = []
        self.flow_buffer = []
        self.img_buffer_flip = []
        self.flow_buffer_flip = []
        self.last_frame = -1

        self.h, self.w, _ = cv2.imread(self.imagefile(1)).shape

    def __getitem__(self, index):
        frame = self.indices[index]

        images = []
        flows = []
        video_tag = 0

        # if there is a new video
        if frame == self.last_frame + 1:
            video_tag = 1
        else:

            video_tag = 0

        self.last_frame = frame

        if video_tag == 0:
            if self.opt.rgb_model != '':
                images = [cv2.imread(self.imagefile(frame + i)).astype(np.float32) for i in range(self.opt.K)]
                images = self.pre_process(images)
                if self.opt.flip_test:
                    self.img_buffer = images[:self.opt.K]
                    self.img_buffer_flip = images[self.opt.K:]
                else:
                    self.img_buffer = images

            if self.opt.pre_extracted_brox_flow and self.opt.flow_model != '':
                flows = [cv2.imread(self.flowfile(min(frame + i, self.nframes))).astype(np.float32) for i in range(self.opt.K + self.opt.ninput - 1)]
                flows = self.pre_process(flows, is_flow=True, ninput=self.opt.ninput)

                if self.opt.flip_test:
                    self.flow_buffer = flows[:self.opt.K]
                    self.flow_buffer_flip = flows[self.opt.K:]
                else:
                    self.flow_buffer = flows

        else:
            if self.opt.rgb_model != '':
                image = cv2.imread(self.imagefile(frame + self.opt.K - 1)).astype(np.float32)
                image, image_flip = self.pre_process_single_frame(image)
                del self.img_buffer[0]
                self.img_buffer.append(image)
                if self.opt.flip_test:
                    del self.img_buffer_flip[0]
                    self.img_buffer_flip.append(image_flip)
                    images = self.img_buffer + self.img_buffer_flip
                else:
                    images = self.img_buffer

            if self.opt.pre_extracted_brox_flow and self.opt.flow_model != '':
                flow = cv2.imread(self.flowfile(min(frame + self.opt.K + self.opt.ninput - 2, self.nframes))).astype(np.float32)
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
        outfile = self.outfile(frame)
        if not os.path.isdir(os.path.dirname(outfile)):
            os.system("mkdir -p '" + os.path.dirname(outfile) + "'")

        return {'outfile': outfile, 'images': images, 'flows': flows, 'meta': {'height': self.h, 'width': self.w, 'output_height': self.output_h, 'output_width': self.output_w}, 'video_tag': video_tag}

    def outfile(self, i):
        return os.path.join(self.opt.inference_dir, "{:0>5}.pkl".format(i))

    def __len__(self):
        return len(self.indices)


def stream_inference(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # torch.backends.cudnn.benchmark = True

    dataset = VisualizationDataset(opt)
    detector = MOCDetector(opt)
    prefetch_dataset = PrefetchDataset(opt, dataset, detector.pre_process, detector.pre_process_single_frame)
    data_loader = torch.utils.data.DataLoader(
        prefetch_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False)

    print('inference begin!', flush=True)
    for iter, data in enumerate(data_loader):

        outfile = data['outfile']

        detections = detector.run(data)

        for i in range(len(outfile)):
            with open(outfile[i], 'wb') as file:
                pickle.dump(detections[i], file)


def det():
    opt = opts().parse()
    if opt.flow_model != "":
        assert 'online flow is not supported yet!'
    os.system("rm -rf " + opt.inference_dir + "/*")
    os.system("rm -rf tmp")
    os.system("mkdir -p '" + os.path.join(opt.inference_dir, 'rgb') + "'")
    os.system("mkdir -p '" + os.path.join(opt.inference_dir, 'flow') + "'")

    video2frames(opt)

    stream_inference(opt)

    build_tubes(opt)

    bbox_dict = pkl_decode(opt)

    vis_bbox(os.path.join(opt.inference_dir, 'rgb'), bbox_dict, opt.instance_level)

    if opt.save_gif:
        rgb2gif(opt.inference_dir)

    rgb2avi(opt.inference_dir)

    print('Finish!', flush=True)


if __name__ == '__main__':
    det()
