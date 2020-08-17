from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import torchvision.models as models
import torch
import torch.nn as nn

import torch.utils.model_zoo as model_zoo
from network.moc_net import MOC_Net
from network.moc_det import MOC_Det, MOC_Backbone


def create_model(arch, branch_info, head_conv, K, flip_test=False):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    model = MOC_Net(arch, num_layers, branch_info, head_conv, K, flip_test=flip_test)
    return model


def create_inference_model(arch, branch_info, head_conv, K, flip_test=False):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    backbone = MOC_Backbone(arch, num_layers)
    branch = MOC_Det(backbone, branch_info, arch, head_conv, K, flip_test=flip_test)
    return backbone, branch


def load_model(model, model_path, optimizer=None, lr=None, ucf_pretrain=False):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if ucf_pretrain:
            if k.startswith('branch.hm') or k.startswith('branch.mov'):
                continue
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    check_state_dict(model.state_dict(), state_dict)
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Resumed optimizer with start lr', lr)
        else:
            print('No optimizer parameters in checkpoint.')

    if 'best' in checkpoint:
        best = checkpoint['best']
    else:
        best = 100
    if optimizer is not None:
        return model, optimizer, start_epoch, best
    else:
        return model


def load_inference_model(backbone, branch, model_path):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]

    backbone.load_state_dict(state_dict, strict=False)

    branch.load_state_dict(state_dict, strict=False)

    return backbone, branch


def save_model(path, model, optimizer=None, epoch=0, best=100):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'best': best,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)


def load_imagenet_pretrained_model(opt, model):
    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        'dla34': 'http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth'
    }
    arch = opt.arch
    if arch == 'dla_34':
        print('load imagenet pretrained dla_34')
        model_url = model_urls['dla34']
        model_weights = model_zoo.load_url(model_url)

    elif arch.startswith('resnet'):
        num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
        assert num_layers in (18, 34, 50, 101, 152)
        arch = arch[:arch.find('_')] if '_' in arch else arch

        print('load imagenet pretrained ', arch)
        url = model_urls['resnet{}'.format(num_layers)]
        model_weights = model_zoo.load_url(url)
        print('=> loading pretrained model {}'.format(url))

    else:
        raise NotImplementedError
    new_state_dict = {}
    for key, value in model_weights.items():
        new_key = 'backbone.base.' + key
        new_state_dict[new_key] = value
    if opt.print_log:
        check_state_dict(model.state_dict(), new_state_dict)
        print('check done!')
    model.load_state_dict(new_state_dict, strict=False)
    if opt.ninput > 1:
        convert2flow(opt.ninput, model)

    return model


def load_coco_pretrained_model(opt, model):
    if opt.arch == 'dla_34':
        print('load coco pretrained dla_34')
        model_path = '../experiment/modelzoo/coco_dla.pth'
    elif opt.arch == 'resnet_18':
        print('load coco pretrained resnet_18')
        model_path = '../experiment/modelzoo/coco_resdcn18.pth'
    elif opt.arch == 'resnet_101':
        print('load coco pretrained resnet_101')
        model_path = '../experiment/modelzoo/coco_resdcn101.pth'
    else:
        raise NotImplementedError
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('wh'):
            new_key = 'branch.' + key
            new_state_dict[new_key] = value
        else:
            new_key = 'backbone.' + key
            new_state_dict[new_key] = value

    if 'resnet' in opt.arch:
        new_state_dict = convert_resnet_dcn(new_state_dict)

    print('load coco pretrained successfully')
    if opt.print_log:
        check_state_dict(model.state_dict(), new_state_dict)
        print('check done!')

    model.load_state_dict(new_state_dict, strict=False)
    if opt.ninput > 1:
        convert2flow(opt.ninput, model)

    return model


def check_state_dict(load_dict, new_dict):
    # check loaded parameters and created model parameters
    for k in new_dict:
        if k in load_dict:
            if new_dict[k].shape != load_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}.'.format(
                          k, load_dict[k].shape, new_dict[k].shape))
                new_dict[k] = load_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in load_dict:
        if not (k in new_dict):
            print('No param {}.'.format(k))
            new_dict[k] = load_dict[k]


def convert2flow(ninput, model):
    modules = list(model.modules())

    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
    # first 7x7 conv : Conv2d(3, 16, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    conv_layer = modules[first_conv_idx]

    container = modules[first_conv_idx - 1]

    # modify parameters, assume the first blob contains the convolution kernels
    params = [x.clone() for x in conv_layer.parameters()]
    # kernel_size: [16, 3, 7, 7]
    kernel_size = params[0].size()
    # new_kernel_size: [16, 3*ninput, 7, 7]
    new_kernel_size = kernel_size[:1] + (3 * ninput, ) + kernel_size[2:]

    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

    new_conv = nn.Conv2d(3 * ninput, conv_layer.out_channels,
                         conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                         bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data  # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

    # replace the first convlution layer

    setattr(container, layer_name, new_conv)
    print('load pretrained model to flow input')
    return model


def convert_resnet_dcn(state_dict):
    new_state_dict = {}
    for k in state_dict:
        if k.startswith('backbone.deconv_layer'):
            new_k = 'backbone.deconv_layer.deconv_layers' + k.split('deconv_layers')[1]
            new_state_dict[new_k] = state_dict[k]
        else:
            new_state_dict[k] = state_dict[k]
    return new_state_dict
