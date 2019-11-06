# -*- coding:utf-8 -*-
# Author:Richard Fang

import numpy as np
import torch


def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.001):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        yy1 = np.maximum(dets[i, 0].numpy(), dets[pos:, 0].numpy())
        xx1 = np.maximum(dets[i, 1].numpy(), dets[pos:, 1].numpy())
        yy2 = np.minimum(dets[i, 2].numpy(), dets[pos:, 2].numpy())
        xx2 = np.minimum(dets[i, 3].numpy(), dets[pos:, 3].numpy())
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep = dets[:, 4][scores > thresh].int()
    print(keep)
    print(scores)

    return keep


boxes = torch.tensor([[200, 200, 400, 400],
                      [220, 220, 420, 420],
                      [200, 240, 400, 440],
                      [240, 200, 440, 400],
                      [1, 1, 2, 2]], dtype=torch.float)
boxscores = torch.tensor([0.8, 0.7, 0.6, 0.5, 0.9], dtype=torch.float)

if torch.cuda.is_available():
    boxes = boxes.cuda()
    boxscores = boxscores.cuda()

soft_nms_pytorch(boxes, boxscores)






