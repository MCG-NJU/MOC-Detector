from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


# feat: b, h*w, 2*K
# ind: b, N


def _gather_feature(feature, index, index_all=None):
    # dim = channel = 2*K
    # feature b, h*w , c
    # index  b, N --> b, N, c
    if index_all is not None:
        index0 = index_all
    else:
        dim = feature.size(2)
        index0 = index.unsqueeze(2).expand(index.size(0), index.size(1), dim)
    feature = feature.gather(1, index0)
    # feature --> b, N, 2*K
    return feature


def _tranpose_and_gather_feature(feature, index, index_all=None):
    # b,c,h,w --> b,h,w,c
    feature = feature.permute(0, 2, 3, 1).contiguous()
    # b,h,w,c --> b,h*w,c
    feature = feature.view(feature.size(0), -1, feature.size(3))
    feature = _gather_feature(feature, index, index_all=index_all)
    # feature --> b, N, 2*K
    return feature


def flip_tensor(x):
    return torch.flip(x, [3])
    # MODIFY for pytorch 0.4.0
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)
