from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import pickle

import numpy as np

from progress.bar import Bar

from datasets.init_dataset import get_dataset

from .ACT_utils import nms2d, nms_tubelets, iou2d


def load_frame_detections(opt, dataset, K, vlist, inference_dir):
    alldets = []  # list of numpy array with <video_index> <frame_index> <ilabel> <score> <x1> <y1> <x2> <y2>
    bar = Bar('{}'.format('FrameAP'), max=len(vlist))
    for iv, v in enumerate(vlist):
        h, w = dataset._resolution[v]

        # aggregate the results for each frame
        vdets = {i: np.empty((0, 6), dtype=np.float32) for i in range(1, 1 + dataset._nframes[v])}  # x1, y1, x2, y2, score, ilabel

        # load results for each starting frame
        for i in range(1, 1 + dataset._nframes[v] - K + 1):
            pkl = os.path.join(inference_dir, v, "{:0>5}.pkl".format(i))
            if not os.path.isfile(pkl):
                print("ERROR: Missing extracted tubelets " + pkl)
                sys.exit()

            with open(pkl, 'rb') as fid:
                dets = pickle.load(fid)

            for label in dets:
                # dets  : {label:  N, 4K+1}
                # 4*K+1 : (x1, y1, x2, y2) * K, score
                tubelets = dets[label]
                labels = np.empty((tubelets.shape[0], 1), dtype=np.int32)
                labels[:, 0] = label - 1
                for k in range(K):
                    vdets[i + k] = np.concatenate((vdets[i + k], np.concatenate((tubelets[:, np.array([4 * k, 1 + 4 * k, 2 + 4 * k, 3 + 4 * k, -1])], labels), axis=1)), axis=0)

        # Perform NMS in each frame
        # vdets : {frame_num:  K*N, 6} ---- x1, x2, y1, y2, score, label
        for i in vdets:
            num_objs = vdets[i].shape[0]
            for ilabel in range(len(dataset.labels)):
                vdets[i] = vdets[i].astype(np.float32)
                a = np.where(vdets[i][:, 5] == ilabel)[0]
                if a.size == 0:
                    continue
                vdets[i][vdets[i][:, 5] == ilabel, :5] = nms2d(vdets[i][vdets[i][:, 5] == ilabel, :5], 0.6)
            # alldets: N,8 --------> ith_video, ith_frame, label, score, x1, x2, y1, y2
            alldets.append(np.concatenate((iv * np.ones((num_objs, 1), dtype=np.float32), i * np.ones((num_objs, 1),
                                                                                                      dtype=np.float32), vdets[i][:, np.array([5, 4, 0, 1, 2, 3], dtype=np.int32)]), axis=1))
        Bar.suffix = '[{0}/{1}]:{2}|Tot: {total:} |ETA: {eta:} '.format(iv + 1, len(vlist), v, total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()
    bar.finish()
    return np.concatenate(alldets, axis=0)


def BuildTubes(opt):
    redo = opt.redo
    if not redo:
        print('load previous linking results...')
        print('if you want to reproduce it, please add --redo')
    Dataset = get_dataset(opt.dataset)
    inference_dirname = opt.inference_dir
    K = opt.K
    split = 'val'
    dataset = Dataset(opt, split)

    print('inference_dirname is ', inference_dirname)
    vlist = dataset._test_videos[opt.split - 1]
    bar = Bar('{}'.format('BuildTubes'), max=len(vlist))
    for iv, v in enumerate(vlist):
        outfile = os.path.join(inference_dirname, v + "_tubes.pkl")
        if os.path.isfile(outfile) and not redo:
            continue

        RES = {}
        nframes = dataset._nframes[v]

        # load detected tubelets
        VDets = {}
        for startframe in range(1, nframes + 2 - K):
            resname = os.path.join(inference_dirname, v, "{:0>5}.pkl".format(startframe))
            if not os.path.isfile(resname):
                print("ERROR: Missing extracted tubelets " + resname)
                sys.exit()

            with open(resname, 'rb') as fid:
                VDets[startframe] = pickle.load(fid)
        for ilabel in range(len(dataset.labels)):
            FINISHED_TUBES = []
            CURRENT_TUBES = []  # tubes is a list of tuple (frame, lstubelets)
            # calculate average scores of tubelets in tubes

            def tubescore(tt):
                return np.mean(np.array([tt[i][1][-1] for i in range(len(tt))]))

            for frame in range(1, dataset._nframes[v] + 2 - K):
                # load boxes of the new frame and do nms while keeping Nkeep highest scored
                ltubelets = VDets[frame][ilabel + 1]  # [:,range(4*K) + [4*K + 1 + ilabel]]  Nx(4K+1) with (x1 y1 x2 y2)*K ilabel-score

                ltubelets = nms_tubelets(ltubelets, 0.6, top_k=10)

                # just start new tubes
                if frame == 1:
                    for i in range(ltubelets.shape[0]):
                        CURRENT_TUBES.append([(1, ltubelets[i, :])])
                    continue

                # sort current tubes according to average score
                avgscore = [tubescore(t) for t in CURRENT_TUBES]
                argsort = np.argsort(-np.array(avgscore))
                CURRENT_TUBES = [CURRENT_TUBES[i] for i in argsort]
                # loop over tubes
                finished = []
                for it, t in enumerate(CURRENT_TUBES):
                    # compute ious between the last box of t and ltubelets
                    last_frame, last_tubelet = t[-1]
                    ious = []
                    offset = frame - last_frame
                    if offset < K:
                        nov = K - offset
                        ious = sum([iou2d(ltubelets[:, 4 * iov:4 * iov + 4], last_tubelet[4 * (iov + offset):4 * (iov + offset + 1)]) for iov in range(nov)]) / float(nov)
                    else:
                        ious = iou2d(ltubelets[:, :4], last_tubelet[4 * K - 4:4 * K])

                    valid = np.where(ious >= 0.5)[0]

                    if valid.size > 0:
                        # take the one with maximum score
                        idx = valid[np.argmax(ltubelets[valid, -1])]
                        CURRENT_TUBES[it].append((frame, ltubelets[idx, :]))
                        ltubelets = np.delete(ltubelets, idx, axis=0)
                    else:
                        if offset >= opt.K:
                            finished.append(it)

                # finished tubes that are done
                for it in finished[::-1]:  # process in reverse order to delete them with the right index why --++--
                    FINISHED_TUBES.append(CURRENT_TUBES[it][:])
                    del CURRENT_TUBES[it]

                # start new tubes
                for i in range(ltubelets.shape[0]):
                    CURRENT_TUBES.append([(frame, ltubelets[i, :])])

            # all tubes are not finished
            FINISHED_TUBES += CURRENT_TUBES

            # build real tubes
            output = []
            for t in FINISHED_TUBES:
                score = tubescore(t)

                # just start new tubes
                if score < 0.005:
                    continue

                beginframe = t[0][0]
                endframe = t[-1][0] + K - 1
                length = endframe + 1 - beginframe

                # delete tubes with short duraton
                if length < 15:
                    continue

                # build final tubes by average the tubelets
                out = np.zeros((length, 6), dtype=np.float32)
                out[:, 0] = np.arange(beginframe, endframe + 1)
                n_per_frame = np.zeros((length, 1), dtype=np.int32)
                for i in range(len(t)):
                    frame, box = t[i]
                    for k in range(K):
                        out[frame - beginframe + k, 1:5] += box[4 * k:4 * k + 4]
                        out[frame - beginframe + k, -1] += box[-1]  # single frame confidence
                        n_per_frame[frame - beginframe + k, 0] += 1
                out[:, 1:] /= n_per_frame
                output.append([out, score])
                # out: [num_frames, (frame idx, x1, y1, x2, y2, score)]

            RES[ilabel] = output
        # RES{ilabel:[(out[length,6],score)]}ilabel[0,...]
        with open(outfile, 'wb') as fid:
            pickle.dump(RES, fid)
        Bar.suffix = '[{0}/{1}]:{2}|Tot: {total:} |ETA: {eta:} '.format(
            iv + 1, len(vlist), v, total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()
    bar.finish()
