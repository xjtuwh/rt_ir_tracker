"""
baseline for Anti-UAV
https://anti-uav.github.io/
"""
from __future__ import absolute_import
import os
import glob
import json
import cv2
import numpy as np

from pytracking.evaluation import Tracker
from pytracking.tracker.base import BaseTracker
from pytracking.tracker.dimp import DiMP
import time
import torch

import argparse
from mmdet.apis import init_detector, inference_detector, show_result
import math


def psr(score_map):
    map = torch.squeeze(score_map)
    peak = torch.max(map).item()
    sum = torch.sum(map).item() - peak
    mean = sum / (map.shape[0] * map.shape[1] - 1)
    map2 = (map - mean) * (map - mean)
    var2 = torch.sum(map2).item() - (peak-mean) * (peak-mean)
    var = math.sqrt(var2) / (map.shape[0] * map.shape[1] - 1)
    return (peak - mean) / var

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',
                default='/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/WH/wh2020/cascade_R_CNN/cascade_rcnn_hrnet_ohem.py',
                        help='train config file path')
    parser.add_argument('--work_dir',
                        default='/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/WH/wh2020/cascade_R_CNN/work_dir',
                        help='the dir to save logs and models')

    args = parser.parse_args()

    return args

def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x,y,w,h.
        bbox2 (numpy.array, list of floats): bounding box in format x,y,w,h.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, w1_1, h1_1) = bbox1
    (x0_2, y0_2, w1_2, h1_2) = bbox2
    x1_1 = x0_1 + w1_1
    x1_2 = x0_2 + w1_2
    y1_1 = y0_1 + h1_1
    y1_2 = y0_2 + h1_2
    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def not_exist(pred):
    return (len(pred) == 1 and pred[0] == 0) or len(pred) == 0


def eval(out_res, label_res):
    measure_per_frame = []
    for _pred, _gt, _exist in zip(out_res, label_res['gt_rect'], label_res['exist']):
        measure_per_frame.append(not_exist(_pred) if not _exist else iou(_pred, _gt) if len(_pred) > 1 else 0)
    return np.mean(measure_per_frame)


def max4(up, down, left, right):
    if up > down:
        if up > left:
            if up > right:
                return up
            else:
                return right
        else:
            if left > right:
                return left
            else:
                return right
    else:
        if down > left:
            if down > right:
                return down
            else:
                return right
        else:
            if left > right:
                return left
            else:
                return right


def main(mode='IR', visulization=False):
    assert mode in ['IR', 'RGB'], 'Only Support IR or RGB to evalute'
    # setup tracker
    output_dir = os.path.join('results', 'dimp_det_challenge')

    # setup experiments
    video_paths = glob.glob(os.path.join('/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/anti_UAV/dataset', 'hard', '*'))    # data path
    video_num = len(video_paths)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    time_all = 0

    # cascade r cnn initilaize
    args = parse_args()
    work_dir = args.work_dir
    config_file = args.config
    checkpoint_file = os.path.join(work_dir, 'epoch_2.pth')
    model = init_detector(config_file, checkpoint_file)

    # run tracking experiments and report performance
    for video_id, video_path in enumerate(video_paths, start=1):
        video_name = os.path.basename(video_path)
        video_file = os.path.join(video_path, '%s.mp4'%mode)
        res_file = os.path.join(video_path, '%s_label.json'%mode)
        with open(res_file, 'r') as f:
            label_res = json.load(f)

        init_rect = label_res['gt_rect'][0]
        capture = cv2.VideoCapture(video_file)

        frame_id = 0
        out_res = []
        th_low = 0.1
        th_pass = 0.8
        # expand_factor = 1
        # expand_factor2 = 2
        seq_time = 0
        while True:
            ret, frame = capture.read()

            if not ret:
                capture.release()
                break
            if frame_id == 0:
                tracker0 = Tracker('dimp', 'dimp18')
                tracker = tracker0.tracker_class(tracker0.get_parameters())
                output = tracker.initialize(frame, init_rect)
                out = init_rect
                out_res.append(init_rect)
                # delta_max = [2.5 * out[2], 2.5 * out[3]]
                max_score_ir = 1
                psr_track = 1

            else:
                tic = time.time()
                output = tracker.track(frame)  # tracking
                out_track = output['target_bbox']
                out = out_track

                # cascade r cnn
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out_det = inference_detector(model, frame)  # detector
                if out_det[0][:, -1].size > 0:
                    idx = np.argmax(out_det[0][:, -1])
                    if out_det[0][idx][-1] >= th_low:
                        x, y, w, h = int(out_det[0][idx][0]), int(out_det[0][idx][1]), int(
                            out_det[0][idx][2] - out_det[0][idx][0]), int(out_det[0][idx][3] - out_det[0][idx][1])

                        sample_pos_det = torch.Tensor([y + (h - 1) / 2, x + (w - 1) / 2])
                        target_sz = torch.Tensor([h, w])
                        search_area = torch.prod(target_sz * tracker.params.search_area_scale).item()
                        target_scale = math.sqrt(search_area) / tracker.img_sample_sz.prod().sqrt()
                        backbone_feat, sample_coords = tracker.extract_backbone_features(tracker.im,
                                                                                         sample_pos_det,
                                                                                         target_scale * tracker.params.scale_factors,
                                                                                         tracker.img_sample_sz)
                        # Extract classification features
                        test_x = tracker.get_classification_features(backbone_feat)
                        # Compute classification scores
                        scores_raw_det = tracker.classify_target(test_x)
                        det = torch.max(scores_raw_det[0]).item()
                        psr_det = psr(scores_raw_det[0])

                        if det > th_pass or (det > max_score_ir and psr_det > psr_track):
                            tracker.pos = sample_pos_det
                            tracker.target_sz = target_sz
                            out = [x, y, w, h]
                        else:
                            if det > max_score_ir or psr_det > psr_track:
                                out = [x, y, w, h]

                else:
                    if max_score_ir < th_low/2:
                        out = np.array([0]).tolist()

                seq_time += (time.time() - tic)
                out_res.append(out)
            if visulization:
                # _gt = label_res['gt_rect'][frame_id]
                # _exist = label_res['exist'][frame_id]
                # if _exist:
                #     cv2.rectangle(frame, (int(_gt[0]), int(_gt[1])), (int(_gt[0] + _gt[2]), int(_gt[1] + _gt[3])),
                #                   (0, 255, 0))
                # cv2.putText(frame, 'exist' if _exist else 'not exist',
                #             (frame.shape[1] // 2 - 20, 30), 1, 2, (0, 255, 0) if _exist else (0, 0, 255), 2)

                cv2.putText(frame, 'Max score = {:.2f}'.format(max_score_ir),
                            (frame.shape[1] // 2 - 60, 50), 1, 2, (255, 0, 0), 2)

                cv2.putText(frame, 'PSR = {:.2f}'.format(psr_track),
                            (frame.shape[1] // 2 - 60, 70), 1, 2, (255, 0, 0), 2)

                if out != [0]:
                    cv2.rectangle(frame, (int(out[0]), int(out[1])), (int(out[0] + out[2]), int(out[1] + out[3])),
                                  (0, 255, 255))
                cv2.imshow(video_name, frame)
                cv2.waitKey(1)
            frame_id += 1
        if visulization:
            cv2.destroyAllWindows()
        seq_avg_time = seq_time / (frame_id - 1)
        time_all += seq_avg_time
        # save result
        output_file = os.path.join(output_dir, '%s_%s.txt' % (video_name, mode))
        with open(output_file, 'w') as f:
            json.dump({'res': out_res}, f)
        print('[%03d/%03d] %20s %5s Fixed Time per frame: %.03f' % (video_id, video_num, video_name, mode, seq_avg_time))

    print('[Overall] %5s Time per frame: %.03f \n' % (
    mode, time_all / video_num))


if __name__ == '__main__':
    main(mode='IR', visulization=False)
