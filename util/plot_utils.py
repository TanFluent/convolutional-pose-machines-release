import numpy as np
import cv2

def _plot_bb(im, bb, cls, scores=[], thres=0.7, textSize=2, textThickness=8):
    h, w, c = im.shape

    for idx, box in enumerate(bb):
        if len(scores) < 1:
            pass
        else:
            if scores[idx] < thres:
                continue
            else:
                pass

        b_w = box[2] * w
        b_h = box[3] * h
        c_x = box[0] * w
        c_y = box[1] * h

        x1 = int(max([0, (c_x - 0.5 * b_w)]))
        x2 = int(min([w, (c_x + 0.5 * b_w)]))
        y1 = int(max([0, (c_y - 0.5 * b_h)]))
        y2 = int(min([h, (c_y + 0.5 * b_h)]))

        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return im, thres