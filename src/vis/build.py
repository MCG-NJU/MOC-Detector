from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import pickle
from vis_dataset import VisualizationDataset

import numpy as np

sys.path.append("..")
from ACT_utils.ACT_utils import nms_tubelets, iou2d


def build_tubes(opt):
    print('inference finish, start building tubes!', flush=True)
    K = opt.K
    dataset = VisualizationDataset(opt)

    outfile = os.path.join(opt.inference_dir, "tubes.pkl")

    RES = {}
    # load detected tubelets
    VDets = {}
    for startframe in range(1, dataset._nframes + 2 - K):
        resname = os.path.join(opt.inference_dir, "{:0>5}.pkl".format(startframe))
        if not os.path.isfile(resname):
            print("ERROR: Missing extracted tubelets " + resname, flush=True)
            sys.exit()

        with open(resname, 'rb') as fid:
            VDets[startframe] = pickle.load(fid)
    for ilabel in range(opt.num_classes):
        FINISHED_TUBES = []
        CURRENT_TUBES = []  # tubes is a list of tuple (frame, lstubelets)
        # calculate average scores of tubelets in tubes

        def tubescore(tt):
            return np.mean(np.array([tt[i][1][-1] for i in range(len(tt))]))

        for frame in range(1, dataset._nframes + 2 - K):
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
    os.system("rm -rf " + opt.inference_dir + "/*.pkl")
    with open(outfile, 'wb') as fid:
        pickle.dump(RES, fid)
