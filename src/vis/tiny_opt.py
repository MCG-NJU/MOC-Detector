from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # basical experiment settings
        self.parser.add_argument('--DATA_ROOT', default='../../data/ucf_test_videos',
                                 help='dataset root path')
        self.parser.add_argument('--inference_dir', default='result_video',
                                 help='vis inference_dir')
        self.parser.add_argument('--vname', default='v_SalsaSpin_g03_c06.mp4',
                                 help='video name')
        self.parser.add_argument('--rgb_model', default='../../experiment/result_model/ucf_dla34_K7_rgb_coco.pth',
                                 help='path to rgb model')
        self.parser.add_argument('--flow_model', default='',
                                 help='path to flow model (online flow extraction is not supported yet)')
        # vis th
        self.parser.add_argument('--tube_vis_th', type=float, default=0.12,
                                 help='min score for visualize tube')
        self.parser.add_argument('--frame_vis_th', type=float, default=0.015,
                                 help='min score for visualize individual frame')
        self.parser.add_argument('--instance_level', action='store_true',
                                 help='draw instance_level action bbox in different color')

        # model seeting
        self.parser.add_argument('--arch', default='dla_34',
                                 help='model architecture. Currently tested'
                                      'resnet_18 | resnet_101 | dla_34')
        self.parser.add_argument('--head_conv', type=int, default=256,
                                 help='conv layer channels for output head'
                                      'default setting is 256 ')
        self.parser.add_argument('--down_ratio', type=int, default=4,
                                 help='output stride. Currently only supports 4.')
        self.parser.add_argument('--K', type=int, default=7,
                                 help='length of action tube')
        self.parser.add_argument('--ninput', type=int, default=5,
                                 help='length of input, 1 for rgb, 5 for flow by default')
        self.parser.add_argument('--num_classes', type=int, default=24,
                                 help='24 num_classes for ucf')

        # dataset seetings
        self.parser.add_argument('--resize_height', type=int, default=288,
                                 help='input image height')
        self.parser.add_argument('--resize_width', type=int, default=288,
                                 help='input image width')

        # inference settings
        self.parser.add_argument('--flip_test', action='store_true',
                                 help='flip data augmentation.')
        self.parser.add_argument('--N', type=int, default=10,
                                 help='max number of output objects.')

        # fusion settings for rgb and flow (online flow extraction is not supported yet)
        self.parser.add_argument('--hm_fusion_rgb', type=float, default=0.5,
                                 help='rgb : th, flow: 1 - th')
        self.parser.add_argument('--mov_fusion_rgb', type=float, default=0.8,
                                 help='rgb : th, flow: 1 - th')
        self.parser.add_argument('--wh_fusion_rgb', type=float, default=0.8,
                                 help='rgb : th, flow: 1 - th')

        # debug
        self.parser.add_argument('--IMAGE_ROOT', default='../../data/ucf24',
                                 help='dataset root path')
        self.parser.add_argument('--pre_extracted_brox_flow', action='store_true',
                                 help='use pre-extracted brox flow and image frames')
        self.parser.add_argument('--save_gif', action='store_true',
                                 help='save uncompressed GIF')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        if opt.flow_model != '':
            opt.pre_extracted_brox_flow = True
        opt.mean = [0.40789654, 0.44719302, 0.47026115]
        opt.std = [0.28863828, 0.27408164, 0.27809835]
        opt.gpus = [0]
        opt.vname = opt.vname.split('_')[1] + '/' + opt.vname
        opt.branch_info = {'hm': opt.num_classes,
                           'mov': 2 * opt.K,
                           'wh': 2 * opt.K}

        return opt
