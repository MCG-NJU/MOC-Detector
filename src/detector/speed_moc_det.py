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
            opt.device = torch.device('cpu')

        self.rgb_model_backbone, self.rgb_model_branch = None, None
        self.flow_model_backbone, self.flow_model_branch = None, None
        self.num_classes = opt.num_classes
        self.opt = opt

    def load_backbone(self):
        opt = self.opt
        if opt.rgb_model != '':
            print('create rgb model')
            self.rgb_model_backbone, self.rgb_model_branch = create_inference_model(opt.arch, opt.branch_info, opt.head_conv, opt.K, flip_test=opt.flip_test)
            self.rgb_model_backbone, self.rgb_model_branch = load_inference_model(self.rgb_model_backbone, self.rgb_model_branch, opt.rgb_model)
            self.rgb_model_backbone = DataParallel(
                self.rgb_model_backbone, device_ids=opt.gpus,
                chunk_sizes=opt.chunk_sizes).to(opt.device)
            self.rgb_model_backbone.eval()
        if opt.flow_model != '':
            print('create flow model')
            self.flow_model_backbone, self.flow_model_branch = create_inference_model(opt.arch, opt.branch_info, opt.head_conv, opt.K, flip_test=opt.flip_test)
            self.flow_model_backbone = convert2flow(opt.ninput, self.flow_model_backbone)
            self.flow_model_backbone, self.flow_model_branch = load_inference_model(self.flow_model_backbone, self.flow_model_branch, opt.flow_model)
            self.flow_model_backbone = DataParallel(
                self.flow_model_backbone, device_ids=opt.gpus,
                chunk_sizes=opt.chunk_sizes).to(opt.device)
            self.flow_model_backbone.eval()

    def load_branch(self):
        opt = self.opt
        if opt.rgb_model != '':
            print('create rgb model')
            self.rgb_model_backbone, self.rgb_model_branch = create_inference_model(opt.arch, opt.branch_info, opt.head_conv, opt.K, flip_test=opt.flip_test)
            self.rgb_model_backbone, self.rgb_model_branch = load_inference_model(self.rgb_model_backbone, self.rgb_model_branch, opt.rgb_model)
            self.rgb_model_branch = DataParallel(
                self.rgb_model_branch, device_ids=opt.gpus,
                chunk_sizes=opt.chunk_sizes).to(opt.device)
            self.rgb_model_branch.eval()
        if opt.flow_model != '':
            print('create flow model')
            self.flow_model_backbone, self.flow_model_branch = create_inference_model(opt.arch, opt.branch_info, opt.head_conv, opt.K, flip_test=opt.flip_test)
            self.flow_model_backbone = convert2flow(opt.ninput, self.flow_model_backbone)
            self.flow_model_backbone, self.flow_model_branch = load_inference_model(self.flow_model_backbone, self.flow_model_branch, opt.flow_model)
            self.flow_model_branch = DataParallel(
                self.flow_model_branch, device_ids=opt.gpus,
                chunk_sizes=opt.chunk_sizes).to(opt.device)
            self.flow_model_branch.eval()

    def pre_process(self, images, is_flow=False, ninput=1):

        images = [cv2.resize(im, (self.opt.resize_height, self.opt.resize_width), interpolation=cv2.INTER_LINEAR) for im in images]

        if self.opt.flip_test:
            data = [np.empty((3 * ninput, self.opt.resize_height, self.opt.resize_width), dtype=np.float32) for i in range(2)]
        else:
            data = [np.empty((3 * ninput, self.opt.resize_height, self.opt.resize_width), dtype=np.float32)]

        mean = np.tile(np.array(self.opt.mean, dtype=np.float32)[:, None, None], (ninput, 1, 1))
        std = np.tile(np.array(self.opt.std, dtype=np.float32)[:, None, None], (ninput, 1, 1))

        for ii in range(ninput):
            data[0][3 * ii:3 * ii + 3, :, :] = np.transpose(images[ii], (2, 0, 1))
            if self.opt.flip_test:
                if is_flow:
                    temp = images[ii].copy()
                    temp = temp[:, ::-1, :]
                    temp[:, :, 2] = 255 - temp[:, :, 2]
                    data[1][3 * ii:3 * ii + 3, :, :] = np.transpose(temp, (2, 0, 1))
                else:
                    data[1][3 * ii:3 * ii + 3, :, :] = np.transpose(images[ii], (2, 0, 1))[:, :, ::-1]
        # normalize
        data[0] = ((data[0] / 255.) - mean) / std
        if self.opt.flip_test:
            data[1] = ((data[1] / 255.) - mean) / std
        return data

    def extract_feature(self, data):

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

        rgb_features, rgb_features_flip, flow_features, flow_features_flip = None, None, None, None
        with torch.no_grad():
            if self.rgb_model_backbone is not None:
                rgb_features = self.rgb_model_backbone(images[0])

                if self.opt.flip_test:
                    rgb_features_flip = self.rgb_model_backbone(images[1])

            if self.flow_model_backbone is not None:
                if self.flow_model_backbone is not None:
                    flow_features = self.flow_model_backbone(flows[0])

                    if self.opt.flip_test:
                        flow_features_flip = self.flow_model_backbone(flows[1])

            return rgb_features, rgb_features_flip, flow_features, flow_features_flip

    def det_process(self, feature):
        with torch.no_grad():
            if self.rgb_model_backbone is not None:
                rgb_output = self.rgb_model_branch(feature['rgb_features'], feature['rgb_features_flip'])
                rgb_hm = rgb_output[0]['hm'].sigmoid_()
                rgb_wh = rgb_output[0]['wh']
                rgb_mov = rgb_output[0]['mov']
                if self.opt.flip_test:
                    rgb_hm_f = rgb_output[1]['hm'].sigmoid_()
                    rgb_wh_f = rgb_output[1]['wh']

                    rgb_hm = (rgb_hm + flip_tensor(rgb_hm_f)) / 2
                    rgb_wh = (rgb_wh + flip_tensor(rgb_wh_f)) / 2

            if self.flow_model_backbone is not None:
                flow_output = self.flow_model_branch(feature['flow_features'], feature['flow_features_flip'])
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
                top_preds[c + 1] = np.concatenate([
                    detections[i, inds, :4 * K].astype(np.float32),
                    detections[i, inds, 4 * K:4 * K + 1].astype(np.float32)], axis=1).tolist()
            results.append(top_preds)

        for i in range(len(results)):
            for j in range(1, self.num_classes + 1):
                results[i][j] = np.array(results[i][j], dtype=np.float32).reshape(-1, self.opt.K * 4 + 1)
        return results

    def run(self, data):
        if self.rgb_model_backbone is not None:
            for i in range(self.opt.K):
                data['rgb_features'][i] = data['rgb_features'][i].to(self.opt.device)
            if self.opt.flip_test:
                for i in range(self.opt.K):
                    data['rgb_features_flip'][i] = data['rgb_features_flip'][i].to(self.opt.device)
        if self.flow_model_backbone is not None:
            for i in range(self.opt.K):
                data['flow_features'][i] = data['flow_features'][i].to(self.opt.device)
            if self.opt.flip_test:
                for i in range(self.opt.K):
                    data['flow_features_flip'][i] = data['flow_features_flip'][i].to(self.opt.device)

        meta = data['meta']
        meta = {k: v.numpy()[0] for k, v in meta.items()}

        # detections--->[b, N, 4*K+1+1]  (bboxes, scores, classes)
        detections = self.det_process(data)

        # detections--->[b, class, 4*K+1]  (bboxes, scores)
        detections = self.post_process(detections, meta['height'], meta['width'],
                                       meta['output_height'], meta['output_width'],
                                       self.opt.num_classes, self.opt.K)

        return detections
