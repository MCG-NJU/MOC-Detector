from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from MOC_utils.utils import _gather_feature, _tranpose_and_gather_feature


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep


def _topN(scores, N=40):
    batch, cat, height, width = scores.size()

    # each class, top N in h*w    [b, c, N]
    topk_scores, topk_index = torch.topk(scores.view(batch, cat, -1), N)

    topk_index = topk_index % (height * width)
    topk_ys = (topk_index / width).int().float()
    topk_xs = (topk_index % width).int().float()

    # cross class, top N    [b, N]
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), N)

    topk_classes = (topk_ind / N).int()
    topk_index = _gather_feature(topk_index.view(batch, -1, 1), topk_ind).view(batch, N)
    topk_ys = _gather_feature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, N)
    topk_xs = _gather_feature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, N)

    return topk_score, topk_index, topk_classes, topk_ys, topk_xs


def moc_decode(heat, wh, mov, N=100, K=5):
    batch, cat, height, width = heat.size()

    # perform 'nms' on heatmaps
    heat = _nms(heat)
    scores, index, classes, ys, xs = _topN(heat, N=N)

    mov = _tranpose_and_gather_feature(mov, index)
    mov = mov.view(batch, N, 2 * K)

    mov_copy = mov.clone()
    mov_copy = mov_copy.view(batch, N, K, 2)
    index_all = torch.zeros((batch, N, K, 2)).cuda()
    xs_all = xs.clone().unsqueeze(2).expand(batch, N, K)
    ys_all = ys.clone().unsqueeze(2).expand(batch, N, K)
    xs_all = xs_all + mov_copy[:, :, :, 0]
    ys_all = ys_all + mov_copy[:, :, :, 1]
    xs_all[:, :, K // 2] = xs
    ys_all[:, :, K // 2] = ys

    xs_all = xs_all.long()
    ys_all = ys_all.long()

    index_all[:, :, :, 0] = xs_all + ys_all * width
    index_all[:, :, :, 1] = xs_all + ys_all * width
    index_all[index_all < 0] = 0
    index_all[index_all > width * height - 1] = width * height - 1
    index_all = index_all.view(batch, N, K * 2).long()

    # gather wh in each location after movement
    wh = _tranpose_and_gather_feature(wh, index, index_all=index_all)
    wh = wh.view(batch, N, 2 * K)

    classes = classes.view(batch, N, 1).float()
    scores = scores.view(batch, N, 1)
    xs = xs.view(batch, N, 1)
    ys = ys.view(batch, N, 1)
    bboxes = []
    for i in range(K):
        bboxes.extend([xs + mov[..., 2 * i:2 * i + 1] - wh[..., 2 * i:2 * i + 1] / 2,
                       ys + mov[..., 2 * i + 1:2 * i + 2] - wh[..., 2 * i + 1:2 * i + 2] / 2,
                       xs + mov[..., 2 * i:2 * i + 1] + wh[..., 2 * i:2 * i + 1] / 2,
                       ys + mov[..., 2 * i + 1:2 * i + 2] + wh[..., 2 * i + 1:2 * i + 2] / 2])
    bboxes = torch.cat(bboxes, dim=2)
    detections = torch.cat([bboxes, scores, classes], dim=2)

    return detections
