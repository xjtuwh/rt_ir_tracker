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
from pytracking import dcf, fourier, TensorList, operation
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor
from pytracking.libs.optimization import GaussNewtonCG, ConjugateGradient, GradientDescentL2
from pytracking.tracker.atom.optim import ConvProblem, FactorizedConvProblem
from pytracking.features import augmentation

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
    return len(pred) == 4 and pred == [0, 0, 0, 0]


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

    tracker = Tracker('atom', 'default')
    output_dir = os.path.join('results', 'atom_expand2_attention')
    output_video_path = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/WH/wh2020/anti-UAV/visualize/' + 'atom_rgbt8/'

    tracker = tracker.tracker_class(tracker.get_parameters())
    tracker_rgb = Tracker('atom', 'default_rgb')
    tracker_rgb = tracker_rgb.tracker_class(tracker_rgb.get_parameters())

    # setup experiments
    video_paths = glob.glob(os.path.join('/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/anti_UAV/dataset', 'test-dev', '*'))    # data path
    video_num = len(video_paths)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_video_path):
        os.makedirs(output_video_path)
    overall_performance = []

    time_all = 0
    modeRGB = 'RGB'

    # run tracking experiments and report performance
    for video_id, video_path in enumerate(video_paths, start=1):
        video_name = os.path.basename(video_path)
        video_file = os.path.join(video_path, '%s.mp4'%mode)
        video_file_RGB = os.path.join(video_path, '%s.mp4' % modeRGB)
        res_file = os.path.join(video_path, '%s_label.json'%mode)
        with open(res_file, 'r') as f:
            label_res = json.load(f)
        res_file_RGB = os.path.join(video_path, '%s_label.json' % modeRGB)
        with open(res_file_RGB, 'r') as f:
            label_res_RGB = json.load(f)

        init_rect = label_res['gt_rect'][0]
        capture = cv2.VideoCapture(video_file)
        init_rect_RGB = label_res_RGB['gt_rect'][0]
        capture_RGB = cv2.VideoCapture(video_file_RGB)

        if visulization:
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(output_video_path + video_name + '.mp4', fourcc, 20.0, (640, 512))

        frame_id = 0
        out_res = []
        th_low = 0.1
        th_pass = 0.8
        expand_factor = 1
        expand_factor2 = 1.5
        # expand_factor_rgb = 5
        seq_time = 0
        while True:
            ret, frame = capture.read()
            ret, frame_RGB = capture_RGB.read()
            if not ret:
                capture.release()
                break
            if frame_id == 0:
                output = tracker.initialize(frame, init_rect)
                output_rgb = tracker_rgb.initialize(frame_RGB, init_rect_RGB)
                out = init_rect
                out_res.append(init_rect)
                delta_max = [2.5 * out[2], 2.5 * out[3]]
                delta_max_rgb = [2.5 * init_rect_RGB[2], 2.5 * init_rect_RGB[3]]
                max_score_ir = 1
                max_score_rgb = 1
            else:
                tic = time.time()
                output = tracker.track(frame)  # tracking
                # output_rgb = tracker_rgb.track(frame_RGB)  # tracking
                out = output['target_bbox']
                # calc delta_max
                delta_max[0] = abs(out[0] - out_res[frame_id-1][0]) if (abs(out[0] - out_res[frame_id-1][0])) > delta_max[0] else delta_max[0]
                delta_max[1] = abs(out[1] - out_res[frame_id-1][1]) if (abs(out[1] - out_res[frame_id-1][1])) > delta_max[1] else delta_max[1]
                if frame_id == 1:
                    max_score_ref = tracker.debug_info['max_score']

                im_rgb = numpy_to_torch(frame_RGB)
                sample_pos_rgb = tracker_rgb.pos.round()
                sample_scales_rgb = tracker_rgb.target_scale * tracker_rgb.params.scale_factors
                test_x_rgb = tracker_rgb.extract_processed_sample(im_rgb, sample_pos_rgb, sample_scales_rgb, tracker_rgb.img_sample_sz)
                scores_raw_rgb = tracker_rgb.apply_filter(test_x_rgb)
                translation_vec_rgb, scale_ind_rgb, s_rgb, flag_rgb = tracker_rgb.localize_target(scores_raw_rgb)

                max_score_rgb = torch.max(scores_raw_rgb[0]).item()
                if flag_rgb != 'not_found':
                    if tracker_rgb.use_iou_net and max_score_rgb > 0.5:
                        update_scale_flag_rgb = getattr(tracker_rgb.params, 'update_scale_when_uncertain',
                                                        True) or flag_rgb != 'uncertain'
                        if getattr(tracker_rgb.params, 'use_classifier', True):
                            tracker_rgb.update_state(sample_pos_rgb + translation_vec_rgb)
                        tracker_rgb.refine_target_box(sample_pos_rgb, sample_scales_rgb[scale_ind_rgb], scale_ind_rgb,
                                                      update_scale_flag_rgb)
                    elif getattr(tracker_rgb.params, 'use_classifier', True):
                        tracker_rgb.update_state(sample_pos_rgb + translation_vec_rgb, sample_scales_rgb[scale_ind_rgb])

                    # calc delta_max
                    delta_max_rgb[0] = abs(tracker_rgb.pos[1] - sample_pos_rgb[1]) if abs(tracker_rgb.pos[1] - sample_pos_rgb[1]) > \
                                                                             delta_max_rgb[0] else delta_max_rgb[0]
                    delta_max_rgb[1] = abs(tracker_rgb.pos[0] - sample_pos_rgb[0]) if abs(tracker_rgb.pos[0] - sample_pos_rgb[0]) > \
                                                                             delta_max_rgb[1] else delta_max_rgb[1]

                if frame_id == 1:
                    max_score_ref_rgb = max_score_rgb
                max_score_ir = tracker.debug_info['max_score']
                if max_score_ir < th_low * 0.5 and max_score_rgb < th_low * 0.5:
                    out = [0, 0, 0, 0]
                if max_score_ir < th_low * max_score_ref:
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
                if max_score_rgb < th_low * max_score_ref_rgb:
                    sample_scales = tracker_rgb.target_scale * tracker_rgb.params.scale_factors
                    sample_pos_left = torch.Tensor(
                        [tracker_rgb.pos[0], tracker_rgb.pos[1] - delta_max_rgb[0] * expand_factor]).round()
                    test_x_left = tracker_rgb.extract_processed_sample(tracker_rgb.im, sample_pos_left, sample_scales, tracker_rgb.img_sample_sz)
                    # Compute scores
                    scores_raw_left = tracker_rgb.apply_filter(test_x_left)
                    left = torch.max(scores_raw_left[0]).item()
                    sample_pos_right = torch.Tensor(
                        [tracker_rgb.pos[0], tracker_rgb.pos[1] + delta_max_rgb[0] * expand_factor]).round()
                    test_x_right = tracker_rgb.extract_processed_sample(tracker_rgb.im, sample_pos_right, sample_scales, tracker_rgb.img_sample_sz)
                    # Compute scores
                    scores_raw_right = tracker_rgb.apply_filter(test_x_right)
                    right = torch.max(scores_raw_right[0]).item()
                    sample_pos_up = torch.Tensor(
                        [tracker_rgb.pos[0] - delta_max_rgb[1] * expand_factor, tracker_rgb.pos[1]]).round()
                    test_x_up = tracker_rgb.extract_processed_sample(tracker_rgb.im, sample_pos_up, sample_scales, tracker_rgb.img_sample_sz)
                    # Compute scores
                    scores_raw_up = tracker_rgb.apply_filter(test_x_up)
                    up = torch.max(scores_raw_up[0]).item()
                    sample_pos_down = torch.Tensor(
                        [tracker_rgb.pos[0] + delta_max_rgb[1] * expand_factor, tracker_rgb.pos[1]]).round()
                    test_x_down = tracker_rgb.extract_processed_sample(tracker_rgb.im, sample_pos_down, sample_scales,
                                                                       tracker_rgb.img_sample_sz)
                    # Compute scores
                    scores_raw_down = tracker_rgb.apply_filter(test_x_down)
                    down = torch.max(scores_raw_down[0]).item()
                    max_response = max4(up, down, left, right)
                    if max_response > th_pass:
                        if max_response == up:
                            translation_vec, scale_ind, s, flag = tracker_rgb.localize_target(scores_raw_up)
                            tracker_rgb.update_state(sample_pos_up + translation_vec, sample_scales[scale_ind])
                        if max_response == down:
                            translation_vec, scale_ind, s, flag = tracker_rgb.localize_target(scores_raw_down)
                            tracker_rgb.update_state(sample_pos_down + translation_vec, sample_scales[scale_ind])
                        if max_response == left:
                            translation_vec, scale_ind, s, flag = tracker_rgb.localize_target(scores_raw_left)
                            tracker_rgb.update_state(sample_pos_left + translation_vec, sample_scales[scale_ind])
                        if max_response == right:
                            translation_vec, scale_ind, s, flag = tracker_rgb.localize_target(scores_raw_right)
                            tracker_rgb.update_state(sample_pos_right + translation_vec, sample_scales[scale_ind])
                    else:
                        sample_pos_left = torch.Tensor(
                            [tracker_rgb.pos[0], tracker_rgb.pos[1] - delta_max_rgb[0] * expand_factor2]).round()
                        test_x_left = tracker_rgb.extract_processed_sample(tracker_rgb.im, sample_pos_left,
                                                                           sample_scales, tracker_rgb.img_sample_sz)
                        # Compute scores
                        scores_raw_left = tracker_rgb.apply_filter(test_x_left)
                        left = torch.max(scores_raw_left[0]).item()
                        sample_pos_right = torch.Tensor(
                            [tracker_rgb.pos[0], tracker_rgb.pos[1] + delta_max_rgb[0] * expand_factor2]).round()
                        test_x_right = tracker_rgb.extract_processed_sample(tracker_rgb.im, sample_pos_right,
                                                                            sample_scales, tracker_rgb.img_sample_sz)
                        # Compute scores
                        scores_raw_right = tracker_rgb.apply_filter(test_x_right)
                        right = torch.max(scores_raw_right[0]).item()
                        sample_pos_up = torch.Tensor(
                            [tracker_rgb.pos[0] - delta_max_rgb[1] * expand_factor2, tracker_rgb.pos[1]]).round()
                        test_x_up = tracker_rgb.extract_processed_sample(tracker_rgb.im, sample_pos_up, sample_scales,
                                                                         tracker_rgb.img_sample_sz)
                        # Compute scores
                        scores_raw_up = tracker_rgb.apply_filter(test_x_up)
                        up = torch.max(scores_raw_up[0]).item()
                        sample_pos_down = torch.Tensor(
                            [tracker_rgb.pos[0] + delta_max_rgb[1] * expand_factor2, tracker_rgb.pos[1]]).round()
                        test_x_down = tracker_rgb.extract_processed_sample(tracker_rgb.im, sample_pos_down,
                                                                           sample_scales,
                                                                           tracker_rgb.img_sample_sz)
                        # Compute scores
                        scores_raw_down = tracker_rgb.apply_filter(test_x_down)
                        down = torch.max(scores_raw_down[0]).item()
                        max_response = max4(up, down, left, right)
                        if max_response > th_pass:
                            if max_response == up:
                                translation_vec, scale_ind, s, flag = tracker_rgb.localize_target(scores_raw_up)
                                tracker_rgb.update_state(sample_pos_up + translation_vec, sample_scales[scale_ind])
                            if max_response == down:
                                translation_vec, scale_ind, s, flag = tracker_rgb.localize_target(scores_raw_down)
                                tracker_rgb.update_state(sample_pos_down + translation_vec, sample_scales[scale_ind])
                            if max_response == left:
                                translation_vec, scale_ind, s, flag = tracker_rgb.localize_target(scores_raw_left)
                                tracker_rgb.update_state(sample_pos_left + translation_vec, sample_scales[scale_ind])
                            if max_response == right:
                                translation_vec, scale_ind, s, flag = tracker_rgb.localize_target(scores_raw_right)
                                tracker_rgb.update_state(sample_pos_right + translation_vec, sample_scales[scale_ind])

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
                cv2.putText(frame, 'Max score IR = {:.2f}'.format(max_score_ir),
                            (frame.shape[1] // 2 - 60, 50), 1, 2, (255, 0, 0), 2)

                cv2.rectangle(frame, (int(out[0]), int(out[1])), (int(out[0] + out[2]), int(out[1] + out[3])),
                              (0, 255, 255))
                cv2.imshow(video_name, frame)
                # out_video.write(frame)
                _gt_rgb = label_res_RGB['gt_rect'][frame_id]
                _exist_rgb = label_res_RGB['exist'][frame_id]
                if _exist_rgb:
                    cv2.rectangle(frame_RGB, (int(_gt_rgb[0]), int(_gt_rgb[1])),
                                  (int(_gt_rgb[0] + _gt_rgb[2]), int(_gt_rgb[1] + _gt_rgb[3])),
                                  (0, 255, 0))
                cv2.putText(frame_RGB, 'exist' if _exist_rgb else 'not exist',
                            (frame_RGB.shape[1] // 2 - 20, 30), 1, 2, (0, 255, 0) if _exist_rgb else (0, 0, 255), 2)

                cv2.putText(frame_RGB, 'Max score RGB = {:.2f}'.format(max_score_rgb),
                            (frame_RGB.shape[1] // 2 - 60, 70), 1, 2, (255, 0, 0), 2)
                new_state_rgb = torch.cat(
                    (tracker_rgb.pos[[1, 0]] - (tracker_rgb.target_sz[[1, 0]] - 1) / 2, tracker_rgb.target_sz[[1, 0]]))
                out_rgb = new_state_rgb.tolist()
                cv2.rectangle(frame_RGB, (int(out_rgb[0]), int(out_rgb[1])),
                              (int(out_rgb[0] + out_rgb[2]), int(out_rgb[1] + out_rgb[3])),
                              (0, 255, 255))
                cv2.imshow(video_name + 'RGB', frame_RGB)
                cv2.waitKey(1)
            frame_id += 1
        if visulization:
            cv2.destroyAllWindows()
            # out_video.release()
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
