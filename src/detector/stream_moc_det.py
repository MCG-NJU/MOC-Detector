from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


import torch

from MOC_utils.model import create_inference_model, load_inference_model, convert2flow
from MOC_utils.data_parallel import DataParallel
from .decode import moc_decode
from MOC_utils.utils import flip_tensor


class MOCDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            assert 'cpu is not supported!'

        self.rgb_model_backbone, self.rgb_model_branch = None, None
        self.flow_model_backbone, self.flow_model_branch = None, None
        if opt.rgb_model != '':
            self.rgb_model_backbone, self.rgb_model_branch = create_inference_model(opt.arch, opt.branch_info, opt.head_conv, opt.K, flip_test=opt.flip_test)
            print('create rgb model', flush=True)
            self.rgb_model_backbone, self.rgb_model_branch = load_inference_model(self.rgb_model_backbone, self.rgb_model_branch, opt.rgb_model)
            print('load rgb model', flush=True)
            self.rgb_model_backbone = DataParallel(
                self.rgb_model_backbone, device_ids=[opt.gpus[0]],
                chunk_sizes=[1]).to(opt.device)
            self.rgb_model_branch = DataParallel(
                self.rgb_model_branch, device_ids=[opt.gpus[0]],
                chunk_sizes=[1]).to(opt.device)
            print('put rgb model to gpu', flush=True)
            self.rgb_model_backbone.eval()
            self.rgb_model_branch.eval()
        if opt.flow_model != '':
            self.flow_model_backbone, self.flow_model_branch = create_inference_model(opt.arch, opt.branch_info, opt.head_conv, opt.K, flip_test=opt.flip_test)
            self.flow_model_backbone = convert2flow(opt.ninput, self.flow_model_backbone)
            print('create flow model', flush=True)
            self.flow_model_backbone, self.flow_model_branch = load_inference_model(self.flow_model_backbone, self.flow_model_branch, opt.flow_model)
            print('load flow model', flush=True)
            self.flow_model_backbone = DataParallel(
                self.flow_model_backbone, device_ids=[opt.gpus[0]],
                chunk_sizes=[1]).to(opt.device)
            self.flow_model_branch = DataParallel(
                self.flow_model_branch, device_ids=[opt.gpus[0]],
                chunk_sizes=[1]).to(opt.device)
            print('put flow model to gpu', flush=True)
            self.flow_model_backbone.eval()
            self.flow_model_branch.eval()

        self.num_classes = opt.num_classes
        self.opt = opt

        self.rgb_buffer = []
        self.flow_buffer = []
        self.rgb_buffer_flip = []
        self.flow_buffer_flip = []

    def pre_process(self, images, is_flow=False, ninput=1):

        K = self.opt.K
        images = [cv2.resize(im, (self.opt.resize_width, self.opt.resize_height), interpolation=cv2.INTER_LINEAR) for im in images]

        if self.opt.flip_test:
            data = [np.empty((3 * ninput, self.opt.resize_height, self.opt.resize_width), dtype=np.float32) for i in range(K * 2)]
        else:
            data = [np.empty((3 * ninput, self.opt.resize_height, self.opt.resize_width), dtype=np.float32) for i in range(K)]

        mean = np.tile(np.array(self.opt.mean, dtype=np.float32)[:, None, None], (ninput, 1, 1))
        std = np.tile(np.array(self.opt.std, dtype=np.float32)[:, None, None], (ninput, 1, 1))

        for i in range(K):
            for ii in range(ninput):
                data[i][3 * ii:3 * ii + 3, :, :] = np.transpose(images[i + ii], (2, 0, 1))
                if self.opt.flip_test:
                    # TODO
                    if is_flow:
                        temp = images[i + ii].copy()
                        temp = temp[:, ::-1, :]
                        temp[:, :, 2] = 255 - temp[:, :, 2]
                        data[i + K][3 * ii:3 * ii + 3, :, :] = np.transpose(temp, (2, 0, 1))
                    else:
                        data[i + K][3 * ii:3 * ii + 3, :, :] = np.transpose(images[i + ii], (2, 0, 1))[:, :, ::-1]
            # normalize
            data[i] = ((data[i] / 255.) - mean) / std
            if self.opt.flip_test:
                data[i + K] = ((data[i + K] / 255.) - mean) / std
        return data

    def pre_process_single_frame(self, images, is_flow=False, ninput=1, data_last=None, data_last_flip=None):
        images = cv2.resize(images, (self.opt.resize_height, self.opt.resize_width), interpolation=cv2.INTER_LINEAR)

        data = np.empty((3 * ninput, self.opt.resize_height, self.opt.resize_width), dtype=np.float32)
        data_flip = np.empty((3 * ninput, self.opt.resize_height, self.opt.resize_width), dtype=np.float32)

        mean = np.array(self.opt.mean, dtype=np.float32)[:, None, None]
        std = np.array(self.opt.std, dtype=np.float32)[:, None, None]
        if not is_flow:
            data = np.transpose(images, (2, 0, 1))
            if self.opt.flip_test:
                data_flip = np.transpose(images, (2, 0, 1))[:, :, ::-1]
            data = ((data / 255.) - mean) / std
            if self.opt.flip_test:
                data_flip = ((data_flip / 255.) - mean) / std

        else:
            data[:3 * ninput - 3, :, :] = data_last[3:, :, :]
            data[3 * ninput - 3:, :, :] = (np.transpose(images, (2, 0, 1)) / 255. - mean) / std
            if self.opt.flip_test:
                temp = images.copy()
                temp = temp[:, ::-1, :]
                temp[:, :, 2] = 255 - temp[:, :, 2]
                data_flip[:3 * ninput - 3, :, :] = data_last_flip[3:, :, :]
                data_flip[3 * ninput - 3:, :, :] = (np.transpose(temp, (2, 0, 1)) / 255. - mean) / std
        return data, data_flip

    def process(self, images, flows, video_tag):
        with torch.no_grad():
            if self.rgb_model_backbone is not None:
                if video_tag == 0:
                    rgb_features = [self.rgb_model_backbone(images[i]) for i in range(self.opt.K)]
                    self.rgb_buffer = rgb_features
                    if self.opt.flip_test:
                        rgb_features_flip = [self.rgb_model_backbone(images[i + self.opt.K]) for i in range(self.opt.K)]
                        self.rgb_buffer_flip = rgb_features_flip
                else:
                    del self.rgb_buffer[0]
                    self.rgb_buffer.append(self.rgb_model_backbone(images[self.opt.K - 1]))
                    if self.opt.flip_test:
                        del self.rgb_buffer_flip[0]
                        self.rgb_buffer_flip.append(self.rgb_model_backbone(images[-1]))
                rgb_output = self.rgb_model_branch(self.rgb_buffer, self.rgb_buffer_flip)
                rgb_hm = rgb_output[0]['hm'].sigmoid_()
                rgb_wh = rgb_output[0]['wh']
                rgb_mov = rgb_output[0]['mov']
                if self.opt.flip_test:
                    rgb_hm_f = rgb_output[1]['hm'].sigmoid_()
                    rgb_wh_f = rgb_output[1]['wh']

                    rgb_hm = (rgb_hm + flip_tensor(rgb_hm_f)) / 2
                    rgb_wh = (rgb_wh + flip_tensor(rgb_wh_f)) / 2

            if self.flow_model_backbone is not None:
                if video_tag == 0:
                    flow_features = [self.flow_model_backbone(flows[i]) for i in range(self.opt.K)]
                    self.flow_buffer = flow_features
                    if self.opt.flip_test:
                        flow_features_flip = [self.flow_model_backbone(flows[i + self.opt.K]) for i in range(self.opt.K)]
                        self.flow_buffer_flip = flow_features_flip
                else:
                    del self.flow_buffer[0]
                    self.flow_buffer.append(self.flow_model_backbone(flows[self.opt.K - 1]))
                    if self.opt.flip_test:
                        del self.flow_buffer_flip[0]
                        self.flow_buffer_flip.append(self.flow_model_backbone(flows[-1]))
                flow_output = self.flow_model_branch(self.flow_buffer, self.flow_buffer_flip)
                flow_hm = flow_output[0]['hm'].sigmoid_()
                flow_wh = flow_output[0]['wh']
                flow_mov = flow_output[0]['mov']
                if self.opt.flip_test:
                    flow_hm_f = flow_output[1]['hm'].sigmoid_()
                    flow_wh_f = flow_output[1]['wh']

                    flow_hm = (flow_hm + flip_tensor(flow_hm_f)) / 2
                    flow_wh = (flow_wh + flip_tensor(flow_wh_f)) / 2

            if self.flow_model_backbone is not None and self.rgb_model_backbone is not None:
                hm = (1 - self.opt.hm_fusion_rgb) * flow_hm + self.opt.hm_fusion_rgb * rgb_hm
                wh = (1 - self.opt.wh_fusion_rgb) * flow_wh + self.opt.wh_fusion_rgb * rgb_wh
                mov = (1 - self.opt.mov_fusion_rgb) * flow_mov + self.opt.mov_fusion_rgb * rgb_mov
            elif self.flow_model_backbone is not None and self.rgb_model_backbone is None:
                hm = flow_hm
                wh = flow_wh
                mov = flow_mov
            elif self.rgb_model_backbone is not None and self.flow_model_backbone is None:
                hm = rgb_hm
                wh = rgb_wh
                mov = rgb_mov
            else:
                print('No model exists.')
                assert 0

            detections = moc_decode(hm, wh, mov, N=self.opt.N, K=self.opt.K)
            return detections

    def post_process(self, detections, height, width, output_height, output_width, num_classes, K):
        detections = detections.detach().cpu().numpy()

        results = []
        for i in range(detections.shape[0]):
            top_preds = {}
            for j in range((detections.shape[2] - 2) // 2):
                # tailor bbox to prevent out of bounds
                detections[i, :, 2 * j] = np.maximum(0, np.minimum(width - 1, detections[i, :, 2 * j] / output_width * width))
                detections[i, :, 2 * j + 1] = np.maximum(0, np.minimum(height - 1, detections[i, :, 2 * j + 1] / output_height * height))
            classes = detections[i, :, -1]
            # gather bbox for each class
            for c in range(self.opt.num_classes):
                inds = (classes == c)
                top_preds[c + 1] = detections[i, inds, :4 * K + 1].astype(np.float32)
            results.append(top_preds)
            return results

    def run(self, data):

        flows = None
        images = None

        if self.rgb_model_backbone is not None:
            images = data['images']
            for i in range(len(images)):
                images[i] = images[i].to(self.opt.device)
        if self.flow_model_backbone is not None:
            flows = data['flows']
            for i in range(len(flows)):
                flows[i] = flows[i].to(self.opt.device)

        meta = data['meta']
        meta = {k: v.numpy()[0] for k, v in meta.items()}

        detections = self.process(images, flows, data['video_tag'])

        detections = self.post_process(detections, meta['height'], meta['width'],
                                       meta['output_height'], meta['output_width'],
                                       self.opt.num_classes, self.opt.K)

        return detections
