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

    tracker = Tracker('dimp', 'dimp18')
    output_dir = os.path.join('results', 'dimp_final')
    tracker = tracker.tracker_class(tracker.get_parameters())

    # setup experiments
    video_paths = glob.glob(os.path.join('/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/anti_UAV/dataset', 'test-challenge', '*'))    # data path
    video_num = len(video_paths)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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
                # print(frame_id)
                # print('not get frame')
                break
            if frame_id == 0:
                output = tracker.initialize(frame, init_rect)
                out = init_rect
                out_res.append(init_rect)
                delta_max = [2.5 * out[2], 2.5 * out[3]]
                max_score_ir = 1
                psr = 1
            else:
                tic = time.time()
                output = tracker.track(frame)  # tracking
                out = output['target_bbox']
                # calc delta_max
                if len(out) > 1 and len(out_res[frame_id - 1]) > 1:
                    delta_max[0] = abs(out[0] - out_res[frame_id - 1][0]) if (abs(out[0] - out_res[frame_id - 1][0])) > \
                                                                         delta_max[0] else delta_max[0]
                    delta_max[1] = abs(out[1] - out_res[frame_id - 1][1]) if (abs(out[1] - out_res[frame_id - 1][1])) > \
                                                                            delta_max[1] else delta_max[1]
                max_score_ir = tracker.debug_info['max_score']
                psr = tracker.psr
                if frame_id == 1:
                    max_score_ref = tracker.debug_info['max_score']
                if tracker.debug_info['max_score'] < th_low * max_score_ref:
                    # sample_scales = tracker.target_scale * tracker.params.scale_factors
                    sample_pos_left = torch.Tensor(
                        [tracker.pos[0], tracker.pos[1] - delta_max[0] * expand_factor]).round()
                    sample_pos_left =  sample_pos_left + ((tracker.feature_sz + tracker.kernel_size) % 2) * tracker.target_scale * \
                                       tracker.img_support_sz / (2 * tracker.feature_sz)
                    backbone_feat, sample_coords = tracker.extract_backbone_features(tracker.im, sample_pos_left,
                                                                                     tracker.target_scale * tracker.params.scale_factors,
                                                                                     tracker.img_sample_sz)
                    # Extract classification features
                    test_x = tracker.get_classification_features(backbone_feat)
                    # Location of sample
                    sample_pos_left, sample_scales_left = tracker.get_sample_location(sample_coords)
                    # Compute classification scores
                    scores_raw_left = tracker.classify_target(test_x)
                    left = torch.max(scores_raw_left[0]).item()

                    sample_pos_right = torch.Tensor(
                        [tracker.pos[0], tracker.pos[1] + delta_max[0] * expand_factor]).round()
                    sample_pos_right =  sample_pos_right + ((tracker.feature_sz + tracker.kernel_size) % 2) * tracker.target_scale * \
                                       tracker.img_support_sz / (2 * tracker.feature_sz)
                    backbone_feat, sample_coords = tracker.extract_backbone_features(tracker.im, sample_pos_right,
                                                                                     tracker.target_scale * tracker.params.scale_factors,
                                                                                     tracker.img_sample_sz)
                    # Extract classification features
                    test_x = tracker.get_classification_features(backbone_feat)
                    # Location of sample
                    sample_pos_right, sample_scales_right = tracker.get_sample_location(sample_coords)
                    # Compute classification scores
                    scores_raw_right = tracker.classify_target(test_x)
                    right = torch.max(scores_raw_right[0]).item()

                    sample_pos_up = torch.Tensor(
                        [tracker.pos[0] - delta_max[1] * expand_factor, tracker.pos[1]]).round()
                    sample_pos_up =  sample_pos_up + ((tracker.feature_sz + tracker.kernel_size) % 2) * tracker.target_scale * \
                                       tracker.img_support_sz / (2 * tracker.feature_sz)
                    backbone_feat, sample_coords = tracker.extract_backbone_features(tracker.im, sample_pos_up,
                                                                                     tracker.target_scale * tracker.params.scale_factors,
                                                                                     tracker.img_sample_sz)
                    # Extract classification features
                    test_x = tracker.get_classification_features(backbone_feat)
                    # Location of sample
                    sample_pos_up, sample_scales_up = tracker.get_sample_location(sample_coords)
                    # Compute classification scores
                    scores_raw_up = tracker.classify_target(test_x)
                    up = torch.max(scores_raw_up[0]).item()

                    sample_pos_down = torch.Tensor(
                        [tracker.pos[0] + delta_max[1] * expand_factor, tracker.pos[1]]).round()
                    sample_pos_down =  sample_pos_down + ((tracker.feature_sz + tracker.kernel_size) % 2) * tracker.target_scale * \
                                       tracker.img_support_sz / (2 * tracker.feature_sz)
                    backbone_feat, sample_coords = tracker.extract_backbone_features(tracker.im, sample_pos_down,
                                                                                     tracker.target_scale * tracker.params.scale_factors,
                                                                                     tracker.img_sample_sz)
                    # Extract classification features
                    test_x = tracker.get_classification_features(backbone_feat)
                    # Location of sample
                    sample_pos_down, sample_scales_down = tracker.get_sample_location(sample_coords)
                    # Compute classification scores
                    scores_raw_down = tracker.classify_target(test_x)
                    down = torch.max(scores_raw_down[0]).item()

                    max_response = max4(up, down, left, right)
                    if max_response > th_pass * max_score_ref:
                        if max_response == up:
                            # Localize the target
                            translation_vec, scale_ind, s, flag = tracker.localize_target(scores_raw_up, sample_scales_up)
                            new_pos = sample_pos_up[scale_ind, :] + translation_vec
                            # tracker.update_state(new_pos, sample_scales_up[scale_ind])
                            if getattr(tracker.params, 'use_iou_net', True):
                                update_scale_flag = getattr(tracker.params, 'update_scale_when_uncertain',
                                                            True) or flag != 'uncertain'
                                if getattr(tracker.params, 'use_classifier', True):
                                    tracker.update_state(new_pos)
                                tracker.refine_target_box(backbone_feat, sample_pos_up[scale_ind, :],
                                                       sample_scales_up[scale_ind], scale_ind, update_scale_flag)

                            elif getattr(tracker.params, 'use_classifier', True):
                                tracker.update_state(new_pos, sample_scales_up[scale_ind])

                        if max_response == down:
                            translation_vec, scale_ind, s, flag = tracker.localize_target(scores_raw_down, sample_scales_down)
                            new_pos = sample_pos_down[scale_ind, :] + translation_vec
                            # tracker.update_state(new_pos, sample_scales_down[scale_ind])
                            if getattr(tracker.params, 'use_iou_net', True):
                                update_scale_flag = getattr(tracker.params, 'update_scale_when_uncertain',
                                                            True) or flag != 'uncertain'
                                if getattr(tracker.params, 'use_classifier', True):
                                    tracker.update_state(new_pos)
                                tracker.refine_target_box(backbone_feat, sample_pos_down[scale_ind, :],
                                                          sample_scales_down[scale_ind], scale_ind, update_scale_flag)

                            elif getattr(tracker.params, 'use_classifier', True):
                                tracker.update_state(new_pos, sample_scales_down[scale_ind])
                        if max_response == left:
                            translation_vec, scale_ind, s, flag = tracker.localize_target(scores_raw_left, sample_scales_left)
                            new_pos = sample_pos_left[scale_ind, :] + translation_vec
                            # tracker.update_state(new_pos, sample_scales_left[scale_ind])
                            if getattr(tracker.params, 'use_iou_net', True):
                                update_scale_flag = getattr(tracker.params, 'update_scale_when_uncertain',
                                                            True) or flag != 'uncertain'
                                if getattr(tracker.params, 'use_classifier', True):
                                    tracker.update_state(new_pos)
                                tracker.refine_target_box(backbone_feat, sample_pos_left[scale_ind, :],
                                                          sample_scales_left[scale_ind], scale_ind, update_scale_flag)

                            elif getattr(tracker.params, 'use_classifier', True):
                                tracker.update_state(new_pos, sample_scales_left[scale_ind])
                        if max_response == right:
                            translation_vec, scale_ind, s, flag = tracker.localize_target(scores_raw_right, sample_scales_right)
                            new_pos = sample_pos_right[scale_ind, :] + translation_vec
                            # tracker.update_state(new_pos, sample_scales_right[scale_ind])
                            if getattr(tracker.params, 'use_iou_net', True):
                                update_scale_flag = getattr(tracker.params, 'update_scale_when_uncertain',
                                                            True) or flag != 'uncertain'
                                if getattr(tracker.params, 'use_classifier', True):
                                    tracker.update_state(new_pos)
                                tracker.refine_target_box(backbone_feat, sample_pos_right[scale_ind, :],
                                                          sample_scales_right[scale_ind], scale_ind, update_scale_flag)

                            elif getattr(tracker.params, 'use_classifier', True):
                                tracker.update_state(new_pos, sample_scales_right[scale_ind])
                        new_state = torch.cat(
                            (tracker.pos[[1, 0]] - (tracker.target_sz[[1, 0]] - 1) / 2, tracker.target_sz[[1, 0]]))
                        out = new_state.tolist()
                    else:
                        # expand again

                        sample_pos_left = torch.Tensor(
                            [tracker.pos[0], tracker.pos[1] - delta_max[0] * expand_factor2]).round()
                        sample_pos_left = sample_pos_left + (
                                    (tracker.feature_sz + tracker.kernel_size) % 2) * tracker.target_scale * \
                                          tracker.img_support_sz / (2 * tracker.feature_sz)
                        backbone_feat, sample_coords = tracker.extract_backbone_features(tracker.im, sample_pos_left,
                                                                                         tracker.target_scale * tracker.params.scale_factors,
                                                                                         tracker.img_sample_sz)
                        # Extract classification features
                        test_x = tracker.get_classification_features(backbone_feat)
                        # Location of sample
                        sample_pos_left, sample_scales_left = tracker.get_sample_location(sample_coords)
                        # Compute classification scores
                        scores_raw_left = tracker.classify_target(test_x)
                        left = torch.max(scores_raw_left[0]).item()

                        sample_pos_right = torch.Tensor(
                            [tracker.pos[0], tracker.pos[1] + delta_max[0] * expand_factor2]).round()
                        sample_pos_right = sample_pos_right + (
                                    (tracker.feature_sz + tracker.kernel_size) % 2) * tracker.target_scale * \
                                           tracker.img_support_sz / (2 * tracker.feature_sz)
                        backbone_feat, sample_coords = tracker.extract_backbone_features(tracker.im, sample_pos_right,
                                                                                         tracker.target_scale * tracker.params.scale_factors,
                                                                                         tracker.img_sample_sz)
                        # Extract classification features
                        test_x = tracker.get_classification_features(backbone_feat)
                        # Location of sample
                        sample_pos_right, sample_scales_right = tracker.get_sample_location(sample_coords)
                        # Compute classification scores
                        scores_raw_right = tracker.classify_target(test_x)
                        right = torch.max(scores_raw_right[0]).item()

                        sample_pos_up = torch.Tensor(
                            [tracker.pos[0] - delta_max[1] * expand_factor2, tracker.pos[1]]).round()
                        sample_pos_up = sample_pos_up + (
                                    (tracker.feature_sz + tracker.kernel_size) % 2) * tracker.target_scale * \
                                        tracker.img_support_sz / (2 * tracker.feature_sz)
                        backbone_feat, sample_coords = tracker.extract_backbone_features(tracker.im, sample_pos_up,
                                                                                         tracker.target_scale * tracker.params.scale_factors,
                                                                                         tracker.img_sample_sz)
                        # Extract classification features
                        test_x = tracker.get_classification_features(backbone_feat)
                        # Location of sample
                        sample_pos_up, sample_scales_up = tracker.get_sample_location(sample_coords)
                        # Compute classification scores
                        scores_raw_up = tracker.classify_target(test_x)
                        up = torch.max(scores_raw_up[0]).item()

                        sample_pos_down = torch.Tensor(
                            [tracker.pos[0] + delta_max[1] * expand_factor2, tracker.pos[1]]).round()
                        sample_pos_down = sample_pos_down + (
                                    (tracker.feature_sz + tracker.kernel_size) % 2) * tracker.target_scale * \
                                          tracker.img_support_sz / (2 * tracker.feature_sz)
                        backbone_feat, sample_coords = tracker.extract_backbone_features(tracker.im, sample_pos_down,
                                                                                         tracker.target_scale * tracker.params.scale_factors,
                                                                                         tracker.img_sample_sz)
                        # Extract classification features
                        test_x = tracker.get_classification_features(backbone_feat)
                        # Location of sample
                        sample_pos_down, sample_scales_down = tracker.get_sample_location(sample_coords)
                        # Compute classification scores
                        scores_raw_down = tracker.classify_target(test_x)
                        down = torch.max(scores_raw_down[0]).item()

                        max_response = max4(up, down, left, right)
                        if max_response > max_score_ref:
                            if max_response == up:
                                # Localize the target
                                translation_vec, scale_ind, s, flag = tracker.localize_target(scores_raw_up,
                                                                                              sample_scales_up)
                                new_pos = sample_pos_up[scale_ind, :] + translation_vec
                                tracker.update_state(new_pos, sample_scales_up[scale_ind])
                            if max_response == down:
                                translation_vec, scale_ind, s, flag = tracker.localize_target(scores_raw_down,
                                                                                              sample_scales_down)
                                new_pos = sample_pos_down[scale_ind, :] + translation_vec
                                tracker.update_state(new_pos, sample_scales_down[scale_ind])
                            if max_response == left:
                                translation_vec, scale_ind, s, flag = tracker.localize_target(scores_raw_left,
                                                                                              sample_scales_left)
                                new_pos = sample_pos_left[scale_ind, :] + translation_vec
                                tracker.update_state(new_pos, sample_scales_left[scale_ind])
                            if max_response == right:
                                translation_vec, scale_ind, s, flag = tracker.localize_target(scores_raw_right,
                                                                                              sample_scales_right)
                                new_pos = sample_pos_right[scale_ind, :] + translation_vec
                                tracker.update_state(new_pos, sample_scales_right[scale_ind])
                            new_state = torch.cat(
                                (tracker.pos[[1, 0]] - (tracker.target_sz[[1, 0]] - 1) / 2, tracker.target_sz[[1, 0]]))
                            out = new_state.tolist()
                        if max_response < th_low and psr < 30:
                            out = np.array([0]).tolist()
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

                cv2.putText(frame, 'Max score = {:.2f}'.format(max_score_ir),
                            (frame.shape[1] // 2 - 60, 50), 1, 2, (255, 0, 0), 2)

                cv2.putText(frame, 'PSR = {:.2f}'.format(psr),
                            (frame.shape[1] // 2 - 60, 70), 1, 2, (255, 0, 0), 2)

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
