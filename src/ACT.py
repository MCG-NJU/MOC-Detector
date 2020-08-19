from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import pickle

import numpy as np

from copy import deepcopy
from datasets.init_dataset import get_dataset

from opts import opts
from ACT_utils.ACT_utils import iou2d, pr_to_ap, nms3dt, iou3dt
from ACT_utils.ACT_build import load_frame_detections, BuildTubes


def frameAP(opt, print_info=True):
    redo = opt.redo
    th = opt.th
    split = 'val'
    model_name = opt.model_name
    Dataset = get_dataset(opt.dataset)
    dataset = Dataset(opt, split)

    inference_dirname = opt.inference_dir
    print('inference_dirname is ', inference_dirname)
    print('threshold is ', th)

    vlist = dataset._test_videos[opt.split - 1]
    # load per-frame detections
    frame_detections_file = os.path.join(inference_dirname, 'frame_detections.pkl')
    if os.path.isfile(frame_detections_file) and not redo:
        print('load previous linking results...')
        print('if you want to reproduce it, please add --redo')
        with open(frame_detections_file, 'rb') as fid:
            alldets = pickle.load(fid)
    else:
        alldets = load_frame_detections(opt, dataset, opt.K, vlist, inference_dirname)
        try:
            with open(frame_detections_file, 'wb') as fid:
                pickle.dump(alldets, fid, protocol=4)
        except:
            print("OverflowError: cannot serialize a bytes object larger than 4 GiB")

    results = {}
    # compute AP for each class
    for ilabel, label in enumerate(dataset.labels):
        # detections of this class
        detections = alldets[alldets[:, 2] == ilabel, :]

        # load ground-truth of this class
        gt = {}
        for iv, v in enumerate(vlist):
            tubes = dataset._gttubes[v]

            if ilabel not in tubes:
                continue

            for tube in tubes[ilabel]:
                for i in range(tube.shape[0]):
                    k = (iv, int(tube[i, 0]))
                    if k not in gt:
                        gt[k] = []
                    gt[k].append(tube[i, 1:5].tolist())

        for k in gt:
            gt[k] = np.array(gt[k])

        # pr will be an array containing precision-recall values
        pr = np.empty((detections.shape[0] + 1, 2), dtype=np.float32)  # precision,recall
        pr[0, 0] = 1.0
        pr[0, 1] = 0.0
        fn = sum([g.shape[0] for g in gt.values()])  # false negatives
        fp = 0  # false positives
        tp = 0  # true positives

        for i, j in enumerate(np.argsort(-detections[:, 3])):
            k = (int(detections[j, 0]), int(detections[j, 1]))
            box = detections[j, 4:8]
            ispositive = False

            if k in gt:
                ious = iou2d(gt[k], box)
                amax = np.argmax(ious)

                if ious[amax] >= th:
                    ispositive = True
                    gt[k] = np.delete(gt[k], amax, 0)

                    if gt[k].size == 0:
                        del gt[k]

            if ispositive:
                tp += 1
                fn -= 1
            else:
                fp += 1

            pr[i + 1, 0] = float(tp) / float(tp + fp)
            pr[i + 1, 1] = float(tp) / float(tp + fn)

        results[label] = pr

    # display results
    ap = 100 * np.array([pr_to_ap(results[label]) for label in dataset.labels])
    frameap_result = np.mean(ap)
    if print_info:
        log_file = open(os.path.join(opt.root_dir, 'result', opt.exp_id), 'a+')
        log_file.write('\nTask_{} frameAP_{}\n'.format(model_name, th))
        print('Task_{} frameAP_{}\n'.format(model_name, th))
        log_file.write("\n{:20s} {:8.2f}\n\n".format("mAP", frameap_result))
        log_file.close()
        print("{:20s} {:8.2f}".format("mAP", frameap_result))

    return frameap_result


def frameAP_error(opt, redo=False):
    th = opt.th
    split = 'val'
    Dataset = get_dataset(opt.dataset)
    dataset = Dataset(opt, split)
    inference_dirname = opt.inference_dir
    print('inference_dirname is ', inference_dirname)
    print('threshold is ', th)
    eval_file = os.path.join(inference_dirname, "frameAP{:g}ErrorAnalysis.pkl".format(th))

    if os.path.isfile(eval_file) and not redo:
        print('load previous linking results...')
        print('if you want to reproduce it, please add --redo')
        with open(eval_file, 'rb') as fid:
            res = pickle.load(fid)
    else:
        vlist = dataset._test_videos[opt.split - 1]
        # load per- frame detections
        frame_detections_file = os.path.join(inference_dirname, 'frame_detections.pkl')
        if os.path.isfile(frame_detections_file) and not redo:
            print('load frameAP pre-result')
            with open(frame_detections_file, 'rb') as fid:
                alldets = pickle.load(fid)
        else:
            alldets = load_frame_detections(opt, dataset, opt.K, vlist, inference_dirname)
            with open(frame_detections_file, 'wb') as fid:
                pickle.dump(alldets, fid)
        res = {}
        # alldets: list of numpy array with <video_index> <frame_index> <ilabel> <score> <x1> <y1> <x2> <y2>
        # compute AP for each class
        print(len(dataset.labels))
        for ilabel, label in enumerate(dataset.labels):
            # detections of this class
            detections = alldets[alldets[:, 2] == ilabel, :]

            gt = {}
            othergt = {}
            labellist = {}

            # iv,v : 0 Basketball/v_Basketball_g01_c01
            for iv, v in enumerate(vlist):
                # tubes: dict {ilabel: (list of)<frame number> <x1> <y1> <x2> <y2>}
                tubes = dataset._gttubes[v]
                # labellist[iv]: label list for v
                labellist[iv] = tubes.keys()

                for il in tubes:
                    # tube: list of <frame number> <x1> <y1> <x2> <y2>
                    for tube in tubes[il]:
                        for i in range(tube.shape[0]):
                            # k: (video_index, frame_index)
                            k = (iv, int(tube[i, 0]))
                            if il == ilabel:
                                if k not in gt:
                                    gt[k] = []
                                gt[k].append(tube[i, 1:5].tolist())
                            else:
                                if k not in othergt:
                                    othergt[k] = []
                                othergt[k].append(tube[i, 1:5].tolist())

            for k in gt:
                gt[k] = np.array(gt[k])
            for k in othergt:
                othergt[k] = np.array(othergt[k])

            dupgt = deepcopy(gt)

            # pr will be an array containing precision-recall values and 4 types of errors:
            # localization, classification, timing, others
            pr = np.empty((detections.shape[0] + 1, 6), dtype=np.float32)  # precision, recall
            pr[0, 0] = 1.0
            pr[0, 1:] = 0.0

            fn = sum([g.shape[0] for g in gt.values()])  # false negatives
            fp = 0  # false positives
            tp = 0  # true positives
            EL = 0  # localization errors
            EC = 0  # classification error: overlap >=0.5 with an another object
            EO = 0  # other errors
            ET = 0  # timing error: the video contains the action but not at this frame

            for i, j in enumerate(np.argsort(-detections[:, 3])):
                k = (int(detections[j, 0]), int(detections[j, 1]))
                box = detections[j, 4:8]
                ispositive = False

                if k in dupgt:
                    if k in gt:
                        ious = iou2d(gt[k], box)
                        amax = np.argmax(ious)
                    if k in gt and ious[amax] >= th:
                        ispositive = True
                        gt[k] = np.delete(gt[k], amax, 0)
                        if gt[k].size == 0:
                            del gt[k]
                    else:
                        EL += 1

                elif k in othergt:
                    ious = iou2d(othergt[k], box)
                    if np.max(ious) >= th:
                        EC += 1
                    else:
                        EO += 1
                elif ilabel in labellist[k[0]]:
                    ET += 1
                else:
                    EO += 1
                if ispositive:
                    tp += 1
                    fn -= 1
                else:
                    fp += 1

                pr[i + 1, 0] = float(tp) / float(tp + fp)  # precision
                pr[i + 1, 1] = float(tp) / float(tp + fn)  # recall
                pr[i + 1, 2] = float(EL) / float(tp + fp)
                pr[i + 1, 3] = float(EC) / float(tp + fp)
                pr[i + 1, 4] = float(ET) / float(tp + fp)
                pr[i + 1, 5] = float(EO) / float(tp + fp)

            res[label] = pr

        # save results
        with open(eval_file, 'wb') as fid:
            pickle.dump(res, fid)

    # display results
    AP = 100 * np.array([pr_to_ap(res[label][:, [0, 1]]) for label in dataset.labels])
    othersap = [100 * np.array([pr_to_ap(res[label][:, [j, 1]]) for label in dataset.labels]) for j in range(2, 6)]

    EL = othersap[0]
    EC = othersap[1]
    ET = othersap[2]
    EO = othersap[3]
    # missed detections = 1 - recall
    EM = 100 - 100 * np.array([res[label][-1, 1] for label in dataset.labels])

    LIST = [AP, EL, EC, ET, EO, EM]

    print('Error Analysis')

    print("")
    print("{:20s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s}".format('label', '   AP   ', '  Loc.  ', '  Cls.  ', '  Time  ', ' Other ', ' missed '))
    print("")
    for il, label in enumerate(dataset.labels):
        print("{:20s} ".format(label) + " ".join(["{:8.2f}".format(L[il]) for L in LIST]))

    print("")
    print("{:20s} ".format("mean") + " ".join(["{:8.2f}".format(np.mean(L)) for L in LIST]))
    print("")


def videoAP(opt, print_info=True):

    th = opt.th
    model_name = opt.model_name
    split = 'val'
    Dataset = get_dataset(opt.dataset)
    dataset = Dataset(opt, split)

    inference_dirname = opt.inference_dir

    vlist = dataset._test_videos[opt.split - 1]
    # load detections
    # alldets = for each label in 1..nlabels, list of tuple (v,score,tube as Kx5 array)
    alldets = {ilabel: [] for ilabel in range(len(dataset.labels))}
    for v in vlist:
        tubename = os.path.join(inference_dirname, v + '_tubes.pkl')
        if not os.path.isfile(tubename):
            print("ERROR: Missing extracted tubes " + tubename)
            sys.exit()

        with open(tubename, 'rb') as fid:
            tubes = pickle.load(fid)
        for ilabel in range(len(dataset.labels)):
            ltubes = tubes[ilabel]
            idx = nms3dt(ltubes, 0.3)
            alldets[ilabel] += [(v, ltubes[i][1], ltubes[i][0]) for i in idx]

    # compute AP for each class
    res = {}
    for ilabel in range(len(dataset.labels)):
        detections = alldets[ilabel]
        # load ground-truth
        gt = {}
        for v in vlist:
            tubes = dataset._gttubes[v]

            if ilabel not in tubes:
                continue

            gt[v] = tubes[ilabel]

            if len(gt[v]) == 0:
                del gt[v]

        # precision,recall
        pr = np.empty((len(detections) + 1, 2), dtype=np.float32)
        pr[0, 0] = 1.0
        pr[0, 1] = 0.0

        fn = sum([len(g) for g in gt.values()])  # false negatives
        fp = 0  # false positives
        tp = 0  # true positives

        for i, j in enumerate(np.argsort(-np.array([dd[1] for dd in detections]))):
            v, score, tube = detections[j]
            ispositive = False

            if v in gt:
                ious = [iou3dt(g, tube) for g in gt[v]]
                amax = np.argmax(ious)
                if ious[amax] >= th:
                    ispositive = True
                    del gt[v][amax]
                    if len(gt[v]) == 0:
                        del gt[v]

            if ispositive:
                tp += 1
                fn -= 1
            else:
                fp += 1

            pr[i + 1, 0] = float(tp) / float(tp + fp)
            pr[i + 1, 1] = float(tp) / float(tp + fn)

        res[dataset.labels[ilabel]] = pr

    # display results
    ap = 100 * np.array([pr_to_ap(res[label]) for label in dataset.labels])
    videoap_result = np.mean(ap)

    if print_info:
        log_file = open(os.path.join(opt.root_dir, 'result', opt.exp_id), 'a+')
        log_file.write('\nTask_{} VideoAP_{}\n'.format(model_name, th))
        print('Task_{} VideoAP_{}\n'.format(opt.model_name, th))
        # for il, _ in enumerate(dataset.labels):
        # print("{:20s} {:8.2f}".format('', ap[il]))
        # log_file.write("{:20s} {:8.2f}\n".format('', ap[il]))
        log_file.write("\n{:20s} {:8.2f}\n\n".format("mAP", videoap_result))
        log_file.close()
        print("{:20s} {:8.2f}".format("mAP", videoap_result))
    return videoap_result


def videpAP_050_095(opt):
    ap = 0
    for i in range(10):
        opt.th = 0.5 + 0.05 * i
        ap += videoAP(opt, print_info=False)
    ap = ap / 10.0
    log_file = open(os.path.join(opt.root_dir, 'result', opt.exp_id), 'a+')
    log_file.write('\nTask_{} VideoAP_0.50:0.95 \n'.format(opt.model_name))
    log_file.write("\n{:20s} {:8.2f}\n\n".format("mAP", ap))
    log_file.close()
    print('Task_{} VideoAP_0.50:0.95 \n'.format(opt.model_name))
    print("\n{:20s} {:8.2f}\n\n".format("mAP", ap))


if __name__ == "__main__":
    opt = opts().parse()
    if not os.path.exists(os.path.join(opt.root_dir, 'result')):
        os.system("mkdir -p '" + os.path.join(opt.root_dir, 'result') + "'")
    if opt.task == 'BuildTubes':
        BuildTubes(opt)
    elif opt.task == 'frameAP':
        frameAP(opt)
    elif opt.task == 'videoAP':
        videoAP(opt)
    elif opt.task == 'frameAP_error':
        frameAP_error(opt)
    elif opt.task == 'videoAP_all':
        videpAP_050_095(opt)
    else:
        raise NotImplementedError('Not implemented:' + opt.task)
