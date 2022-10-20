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
from pytracking.tracker.atom import ATOM

import time
import torch
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
    return len(pred) == 1 and pred == 0


def eval(out_res, label_res):
    measure_per_frame = []
    for _pred, _gt, _exist in zip(out_res, label_res['gt_rect'], label_res['exist']):
        measure_per_frame.append(not_exist(_pred) if not _exist else iou(_pred, _gt))
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
    output_dir = os.path.join('results', 'atom_final')

    # setup experiments
    video_paths = glob.glob(os.path.join('/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/anti_UAV/dataset', 'test-dev', '*'))    # data path
    video_num = len(video_paths)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    overall_performance = []

    time_all = 0
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
        expand_factor = 1
        expand_factor2 = 2
        seq_time = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                capture.release()
                break
            if frame_id == 0:
                tracker0 = Tracker('atom', 'default')
                tracker = tracker0.tracker_class(tracker0.get_parameters())
                output = tracker.initialize(frame, init_rect)
                out = init_rect
                out_res.append(init_rect)
                delta_max = [2.5 * out[2], 2.5 * out[3]]
            else:
                tic = time.time()
                output = tracker.track(frame)  # tracking
                out = output['target_bbox']
                # calc delta_max
                delta_max[0] = abs(out[0] - out_res[frame_id-1][0]) if (abs(out[0] - out_res[frame_id-1][0])) > delta_max[0] else delta_max[0]
                delta_max[1] = abs(out[1] - out_res[frame_id-1][1]) if (abs(out[1] - out_res[frame_id-1][1])) > delta_max[1] else delta_max[1]
                if frame_id == 1:
                    max_score_ref = tracker.debug_info['max_score']
                if tracker.debug_info['max_score'] < th_low * max_score_ref:
                    sample_scales = tracker.target_scale * tracker.params.scale_factors
                    sample_pos_left = torch.Tensor(
                        [tracker.pos[0], tracker.pos[1] - delta_max[0] * expand_factor]).round()
                    test_x_left = tracker.extract_processed_sample(tracker.im, sample_pos_left, sample_scales, tracker.img_sample_sz)
                    # Compute scores
                    scores_raw_left = tracker.apply_filter(test_x_left)
                    left = torch.max(scores_raw_left[0]).item()
                    sample_pos_right = torch.Tensor(
                        [tracker.pos[0], tracker.pos[1] + delta_max[0] * expand_factor]).round()
                    test_x_right = tracker.extract_processed_sample(tracker.im, sample_pos_right, sample_scales, tracker.img_sample_sz)
                    # Compute scores
                    scores_raw_right = tracker.apply_filter(test_x_right)
                    right = torch.max(scores_raw_right[0]).item()
                    sample_pos_up = torch.Tensor(
                        [tracker.pos[0] - delta_max[1] * expand_factor, tracker.pos[1]]).round()
                    test_x_up = tracker.extract_processed_sample(tracker.im, sample_pos_up, sample_scales, tracker.img_sample_sz)
                    # Compute scores
                    scores_raw_up = tracker.apply_filter(test_x_up)
                    up = torch.max(scores_raw_up[0]).item()
                    sample_pos_down = torch.Tensor(
                        [tracker.pos[0] + delta_max[1] * expand_factor, tracker.pos[1]]).round()
                    test_x_down = tracker.extract_processed_sample(tracker.im, sample_pos_down, sample_scales,
                                                                 tracker.img_sample_sz)
                    # Compute scores
                    scores_raw_down = tracker.apply_filter(test_x_down)
                    down = torch.max(scores_raw_down[0]).item()
                    max_response = max4(up, down, left, right)
                    if max_response > th_pass * max_score_ref:
                        if max_response == up:
                            translation_vec, scale_ind, s, flag = tracker.localize_target(scores_raw_up)
                            tracker.update_state(sample_pos_up + translation_vec, sample_scales[scale_ind])
                        if max_response == down:
                            translation_vec, scale_ind, s, flag = tracker.localize_target(scores_raw_down)
                            tracker.update_state(sample_pos_down + translation_vec, sample_scales[scale_ind])
                        if max_response == left:
                            translation_vec, scale_ind, s, flag = tracker.localize_target(scores_raw_left)
                            tracker.update_state(sample_pos_left + translation_vec, sample_scales[scale_ind])
                        if max_response == right:
                            translation_vec, scale_ind, s, flag = tracker.localize_target(scores_raw_right)
                            tracker.update_state(sample_pos_right + translation_vec, sample_scales[scale_ind])
                        new_state = torch.cat(
                            (tracker.pos[[1, 0]] - (tracker.target_sz[[1, 0]] - 1) / 2, tracker.target_sz[[1, 0]]))
                        out = new_state.tolist()
                    else:
                        # expand again
                        sample_pos_left = torch.Tensor(
                            [tracker.pos[0], tracker.pos[1] - delta_max[0] * expand_factor2]).round()
                        test_x_left = tracker.extract_processed_sample(tracker.im, sample_pos_left, sample_scales,
                                                                       tracker.img_sample_sz)
                        # Compute scores
                        scores_raw_left = tracker.apply_filter(test_x_left)
                        left = torch.max(scores_raw_left[0]).item()
                        sample_pos_right = torch.Tensor(
                            [tracker.pos[0], tracker.pos[1] + delta_max[0] * expand_factor2]).round()
                        test_x_right = tracker.extract_processed_sample(tracker.im, sample_pos_right, sample_scales,
                                                                        tracker.img_sample_sz)
                        # Compute scores
                        scores_raw_right = tracker.apply_filter(test_x_right)
                        right = torch.max(scores_raw_right[0]).item()
                        sample_pos_up = torch.Tensor(
                            [tracker.pos[0] - delta_max[1] * expand_factor2, tracker.pos[1]]).round()
                        test_x_up = tracker.extract_processed_sample(tracker.im, sample_pos_up, sample_scales,
                                                                     tracker.img_sample_sz)
                        # Compute scores
                        scores_raw_up = tracker.apply_filter(test_x_up)
                        up = torch.max(scores_raw_up[0]).item()
                        sample_pos_down = torch.Tensor(
                            [tracker.pos[0] + delta_max[1] * expand_factor2, tracker.pos[1]]).round()
                        test_x_down = tracker.extract_processed_sample(tracker.im, sample_pos_down, sample_scales,
                                                                       tracker.img_sample_sz)
                        # Compute scores
                        scores_raw_down = tracker.apply_filter(test_x_down)
                        down = torch.max(scores_raw_down[0]).item()
                        max_response = max4(up, down, left, right)
                        if max_response > th_pass:
                            if max_response == up:
                                translation_vec, scale_ind, s, flag = tracker.localize_target(scores_raw_up)
                                tracker.update_state(sample_pos_up + translation_vec, sample_scales[scale_ind])
                            if max_response == down:
                                translation_vec, scale_ind, s, flag = tracker.localize_target(scores_raw_down)
                                tracker.update_state(sample_pos_down + translation_vec, sample_scales[scale_ind])
                            if max_response == left:
                                translation_vec, scale_ind, s, flag = tracker.localize_target(scores_raw_left)
                                tracker.update_state(sample_pos_left + translation_vec, sample_scales[scale_ind])
                            if max_response == right:
                                translation_vec, scale_ind, s, flag = tracker.localize_target(scores_raw_right)
                                tracker.update_state(sample_pos_right + translation_vec, sample_scales[scale_ind])
                            new_state = torch.cat(
                                (tracker.pos[[1, 0]] - (tracker.target_sz[[1, 0]] - 1) / 2, tracker.target_sz[[1, 0]]))
                            out = new_state.tolist()
                seq_time += (time.time() - tic)
                out_res.append(out)
            if visulization:
                _gt = label_res['gt_rect'][frame_id]
                _exist = label_res['exist'][frame_id]
                if _exist:
                    cv2.rectangle(frame, (int(_gt[0]), int(_gt[1])), (int(_gt[0] + _gt[2]), int(_gt[1] + _gt[3])),
                                  (0, 255, 0))
                cv2.putText(frame, 'exist' if _exist else 'not exist',
                            (frame.shape[1] // 2 - 20, 30), 1, 2, (0, 255, 0) if _exist else (0, 0, 255), 2)

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

        mixed_measure = eval(out_res, label_res)
        overall_performance.append(mixed_measure)
        print('[%03d/%03d] %20s %5s Fixed Measure: %.03f Time per frame: %.03f' % (video_id, video_num, video_name, mode, mixed_measure, seq_avg_time))

    print('[Overall] %5s Mixed Measure: %.03f Time per frame: %.03f \n' % (mode, np.mean(overall_performance), time_all / video_num))


if __name__ == '__main__':
    main(mode='IR', visulization=False)
