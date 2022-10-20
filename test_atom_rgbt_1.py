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
import torch.nn.functional as F
import torch.nn
import math
import time
from pytracking import dcf, fourier, TensorList, operation
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor
from pytracking.libs.optimization import GaussNewtonCG, ConjugateGradient, GradientDescentL2
from pytracking.tracker.atom.optim import ConvProblem, FactorizedConvProblem
from pytracking.features import augmentation


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


def main(mode='IR', visulization=False):
    assert mode in ['IR', 'RGB'], 'Only Support IR and/or RGB to evalute'
    output_dir = os.path.join('results', 'atom_rgbt_1')
    output_video_path = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/WH/wh2020/anti-UAV/visualize/' + 'atom_rgbt_1/'
    # setup tracker
    tracker_ir = Tracker('atom', 'default')
    tracker_ir = tracker_ir.tracker_class(tracker_ir.get_parameters())
    tracker_rgb = Tracker('atom', 'default_rgb')
    tracker_rgb = tracker_rgb.tracker_class(tracker_rgb.get_parameters())

    # setup experiments
    video_paths = glob.glob(os.path.join('/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/anti_UAV/dataset', 'test-dev', '*'))    # data path test-dev
    video_num = len(video_paths)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_video_path):
        os.makedirs(output_video_path)
    overall_performance = []

    modeIR = 'IR'
    modeRGB = 'RGB'

    # run tracking experiments and report performance
    for video_id, video_path in enumerate(video_paths, start=1):
        video_name = os.path.basename(video_path)
        video_file_IR = os.path.join(video_path, '%s.mp4'%modeIR)
        video_file_RGB = os.path.join(video_path, '%s.mp4' % modeRGB)
        res_file_IR = os.path.join(video_path, '%s_label.json'%modeIR)
        with open(res_file_IR, 'r') as f:
            label_res_IR = json.load(f)
        res_file_RGB = os.path.join(video_path, '%s_label.json'%modeRGB)
        with open(res_file_RGB, 'r') as f:
            label_res_RGB = json.load(f)

        init_rect_IR = label_res_IR['gt_rect'][0]
        capture_IR = cv2.VideoCapture(video_file_IR)
        init_rect_RGB = label_res_RGB['gt_rect'][0]
        capture_RGB = cv2.VideoCapture(video_file_RGB)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path + video_name + '.mp4', fourcc, 20.0, (640, 512))

        frame_id = 0
        out_res = []

        th_pass = 0.8
        expand_factor = 2.5
        while True:
            ret, frame_IR = capture_IR.read()
            ret, frame_RGB = capture_RGB.read()
            if not ret:
                capture_IR.release()
                capture_RGB.release()
                break
            if frame_id == 0:
                output_ir = tracker_ir.initialize(frame_IR, init_rect_IR)
                output_rgb = tracker_rgb.initialize(frame_RGB, init_rect_RGB)
                new_state = init_rect_IR
                out_res.append(init_rect_IR)
                max_score_ir = 1
                max_score_rgb = 1
            else:
                # ------------ IR Tracking -----------------#
                # tracker_ir.frame_num += 1
                im_ir = numpy_to_torch(frame_IR)
                tracker_ir.im = im_ir
                sample_pos_ir = tracker_ir.pos.round()
                sample_scales_ir = tracker_ir.target_scale * tracker_ir.params.scale_factors
                test_x_ir = tracker_ir.extract_processed_sample(im_ir, tracker_ir.pos, sample_scales_ir, tracker_ir.img_sample_sz)
                # Compute scores
                scores_raw_ir = tracker_ir.apply_filter(test_x_ir)

                # ------------ RGB Tracking ---------------#
                # tracker_rgb.frame_num += 1
                im_rgb = numpy_to_torch(frame_RGB)
                tracker_rgb.im = im_rgb
                sample_pos_rgb = tracker_rgb.pos.round()
                sample_scales_rgb = tracker_rgb.target_scale * tracker_rgb.params.scale_factors
                test_x_rgb = tracker_rgb.extract_processed_sample(im_rgb, tracker_rgb.pos, sample_scales_rgb, tracker_rgb.img_sample_sz)
                # Compute scores
                scores_raw_rgb = tracker_rgb.apply_filter(test_x_rgb)

                translation_vec_ir, scale_ind_ir, s_ir, flag_ir = tracker_ir.localize_target(scores_raw_ir* 0.9 + scores_raw_rgb * 0.1)   #
                translation_vec_rgb, scale_ind_rgb, s_rgb, flag_rgb = tracker_rgb.localize_target(scores_raw_rgb* 0.9 + scores_raw_ir * 0.1)  #

                if flag_ir == 'not_found' and flag_rgb == 'not_found':
                    out_res.append([0, 0, 0, 0])
                else:
                    if flag_ir != 'not_found':
                        if tracker_ir.use_iou_net:
                            update_scale_flag_ir = getattr(tracker_ir.params, 'update_scale_when_uncertain',
                                                        True) or flag_ir != 'uncertain'
                            if getattr(tracker_ir.params, 'use_classifier', True):
                                tracker_ir.update_state(sample_pos_ir + translation_vec_ir)
                            tracker_ir.refine_target_box(sample_pos_ir, sample_scales_ir[scale_ind_ir], scale_ind_ir, update_scale_flag_ir)
                        elif getattr(tracker_ir.params, 'use_classifier', True):
                            tracker_ir.update_state(sample_pos_ir + translation_vec_ir, sample_scales_ir[scale_ind_ir])
                    if flag_rgb != 'not_found':
                        if tracker_rgb.use_iou_net:
                            update_scale_flag_rgb = getattr(tracker_rgb.params, 'update_scale_when_uncertain',
                                                           True) or flag_rgb != 'uncertain'
                            if getattr(tracker_rgb.params, 'use_classifier', True):
                                tracker_rgb.update_state(sample_pos_rgb + translation_vec_rgb)
                            tracker_rgb.refine_target_box(sample_pos_rgb, sample_scales_rgb[scale_ind_rgb], scale_ind_rgb,
                                                         update_scale_flag_rgb)
                        elif getattr(tracker_rgb.params, 'use_classifier', True):
                            tracker_rgb.update_state(sample_pos_rgb + translation_vec_rgb, sample_scales_rgb[scale_ind_rgb])

                    score_map_ir = s_ir[scale_ind_ir, ...]
                    max_score_ir = torch.max(score_map_ir).item()
                    score_map_rgb = s_rgb[scale_ind_rgb, ...]
                    max_score_rgb = torch.max(score_map_rgb).item()

                    # ------- UPDATE ------- #
                    # Check flags and set learning rate if hard negative
                    update_flag = flag_ir not in ['not_found', 'uncertain']
                    hard_negative = (flag_ir == 'hard_negative')
                    learning_rate = tracker_ir.params.hard_negative_learning_rate if hard_negative else None
                    if update_flag:
                        # Get train sample
                        train_x = TensorList([x[scale_ind_ir:scale_ind_ir + 1, ...] for x in test_x_ir])

                        # Create label for sample
                        train_y = tracker_ir.get_label_function(sample_pos_ir, sample_scales_ir[scale_ind_ir])

                        # Update memory
                        tracker_ir.update_memory(train_x, train_y, learning_rate)

                    if hard_negative:
                        tracker_ir.filter_optimizer.run(tracker_ir.params.hard_negative_CG_iter)
                    elif frame_id % tracker_ir.params.train_skipping == 0:
                        tracker_ir.filter_optimizer.run(tracker_ir.params.CG_iter)

                    # Set the pos of the tracker to iounet pos
                    if tracker_ir.use_iou_net and flag_ir != 'not_found':
                        tracker_ir.pos = tracker_ir.pos_iounet.clone()
                    if tracker_rgb.use_iou_net and flag_rgb != 'not_found':
                        tracker_rgb.pos = tracker_rgb.pos_iounet.clone()

                    # Return new state
                    new_state = torch.cat(
                        (tracker_ir.pos[[1, 0]] - (tracker_ir.target_sz[[1, 0]] - 1) / 2, tracker_ir.target_sz[[1, 0]]))
                    out_res.append(new_state.tolist())
                if torch.max(scores_raw_ir[0]).item() < 0.1:
                    sample_pos_ir_left = torch.Tensor([tracker_ir.pos[0] - (tracker_ir.target_sz[0])*expand_factor, tracker_ir.pos[1]]).round()
                    test_x_ir_left = tracker_ir.extract_processed_sample(im_ir, sample_pos_ir_left, sample_scales_ir, tracker_ir.img_sample_sz)
                    # Compute scores
                    scores_raw_ir_left = tracker_ir.apply_filter(test_x_ir_left)
                    left = torch.max(scores_raw_ir_left[0]).item()
                    sample_pos_ir_right = torch.Tensor([tracker_ir.pos[0] + (tracker_ir.target_sz[0])*expand_factor, tracker_ir.pos[1]]).round()
                    test_x_ir_right = tracker_ir.extract_processed_sample(im_ir, sample_pos_ir_right, sample_scales_ir, tracker_ir.img_sample_sz)
                    # Compute scores
                    scores_raw_ir_right = tracker_ir.apply_filter(test_x_ir_right)
                    right = torch.max(scores_raw_ir_right[0]).item()
                    sample_pos_ir_bottom = torch.Tensor([tracker_ir.pos[0], tracker_ir.pos[1] - (tracker_ir.target_sz[1])*expand_factor]).round()
                    test_x_ir_bottom = tracker_ir.extract_processed_sample(im_ir, sample_pos_ir_bottom, sample_scales_ir, tracker_ir.img_sample_sz)
                    # Compute scores
                    scores_raw_ir_bottom = tracker_ir.apply_filter(test_x_ir_bottom)
                    bottom = torch.max(scores_raw_ir_bottom[0]).item()
                    sample_pos_ir_up = torch.Tensor([tracker_ir.pos[0], tracker_ir.pos[1] + (tracker_ir.target_sz[1])*expand_factor]).round()
                    test_x_ir_up = tracker_ir.extract_processed_sample(im_ir, sample_pos_ir_up, sample_scales_ir, tracker_ir.img_sample_sz)
                    # Compute scores
                    scores_raw_ir_up = tracker_ir.apply_filter(test_x_ir_up)
                    up = torch.max(scores_raw_ir_up[0]).item()

                    if left > right:
                        if left > bottom:
                            if left > up:
                                if left > th_pass:
                                    #  tHIS IS THE REAL TRAGET PROCESS THE LOCATION
                                    translation_vec_ir, scale_ind_ir, s_ir, flag_ir = tracker_ir.localize_target(scores_raw_ir_left)
                                    if tracker_ir.use_iou_net:
                                        update_scale_flag_ir = getattr(tracker_ir.params, 'update_scale_when_uncertain',
                                                                       True) or flag_ir != 'uncertain'
                                        if getattr(tracker_ir.params, 'use_classifier', True):
                                            tracker_ir.update_state(sample_pos_ir_left + translation_vec_ir)
                                        tracker_ir.refine_target_box(sample_pos_ir_left, sample_scales_ir[scale_ind_ir],
                                                                     scale_ind_ir, update_scale_flag_ir)
                                    elif getattr(tracker_ir.params, 'use_classifier', True):
                                        tracker_ir.update_state(sample_pos_ir_left + translation_vec_ir,
                                                                sample_scales_ir[scale_ind_ir])

                            else:
                                if up > th_pass:
                                    # process
                                    translation_vec_ir, scale_ind_ir, s_ir, flag_ir = tracker_ir.localize_target(scores_raw_ir_up)
                                    if tracker_ir.use_iou_net:
                                        update_scale_flag_ir = getattr(tracker_ir.params, 'update_scale_when_uncertain',
                                                                       True) or flag_ir != 'uncertain'
                                        if getattr(tracker_ir.params, 'use_classifier', True):
                                            tracker_ir.update_state(sample_pos_ir_up + translation_vec_ir)
                                        tracker_ir.refine_target_box(sample_pos_ir_up, sample_scales_ir[scale_ind_ir],
                                                                     scale_ind_ir, update_scale_flag_ir)
                                    elif getattr(tracker_ir.params, 'use_classifier', True):
                                        tracker_ir.update_state(sample_pos_ir_up + translation_vec_ir,
                                                                sample_scales_ir[scale_ind_ir])

                        else:
                            if bottom > up:
                                if bottom > th_pass:
                                    #  tHIS IS THE REAL TRAGET PROCESS THE LOCATION
                                    translation_vec_ir, scale_ind_ir, s_ir, flag_ir = tracker_ir.localize_target(scores_raw_ir_bottom)
                                    if tracker_ir.use_iou_net:
                                        update_scale_flag_ir = getattr(tracker_ir.params, 'update_scale_when_uncertain',
                                                                       True) or flag_ir != 'uncertain'
                                        if getattr(tracker_ir.params, 'use_classifier', True):
                                            tracker_ir.update_state(sample_pos_ir_bottom + translation_vec_ir)
                                        tracker_ir.refine_target_box(sample_pos_ir_bottom, sample_scales_ir[scale_ind_ir],
                                                                     scale_ind_ir, update_scale_flag_ir)
                                    elif getattr(tracker_ir.params, 'use_classifier', True):
                                        tracker_ir.update_state(sample_pos_ir_bottom + translation_vec_ir,
                                                                sample_scales_ir[scale_ind_ir])

                            else:
                                if up > th_pass:
                                    # process
                                    translation_vec_ir, scale_ind_ir, s_ir, flag_ir = tracker_ir.localize_target(
                                        scores_raw_ir_up)
                                    if tracker_ir.use_iou_net:
                                        update_scale_flag_ir = getattr(tracker_ir.params, 'update_scale_when_uncertain',
                                                                       True) or flag_ir != 'uncertain'
                                        if getattr(tracker_ir.params, 'use_classifier', True):
                                            tracker_ir.update_state(sample_pos_ir_up + translation_vec_ir)
                                        tracker_ir.refine_target_box(sample_pos_ir_up, sample_scales_ir[scale_ind_ir],
                                                                     scale_ind_ir, update_scale_flag_ir)
                                    elif getattr(tracker_ir.params, 'use_classifier', True):
                                        tracker_ir.update_state(sample_pos_ir_up + translation_vec_ir,
                                                                sample_scales_ir[scale_ind_ir])

                    else:
                        if right > bottom:
                            if right > up:
                                if right > th_pass:
                                    #  tHIS IS THE REAL TRAGET PROCESS THE LOCATION
                                    translation_vec_ir, scale_ind_ir, s_ir, flag_ir = tracker_ir.localize_target(
                                        scores_raw_ir_right)
                                    if tracker_ir.use_iou_net:
                                        update_scale_flag_ir = getattr(tracker_ir.params, 'update_scale_when_uncertain',
                                                                       True) or flag_ir != 'uncertain'
                                        if getattr(tracker_ir.params, 'use_classifier', True):
                                            tracker_ir.update_state(sample_pos_ir_right + translation_vec_ir)
                                        tracker_ir.refine_target_box(sample_pos_ir_right, sample_scales_ir[scale_ind_ir],
                                                                     scale_ind_ir, update_scale_flag_ir)
                                    elif getattr(tracker_ir.params, 'use_classifier', True):
                                        tracker_ir.update_state(sample_pos_ir_right + translation_vec_ir,
                                                                sample_scales_ir[scale_ind_ir])

                            else:
                                if up > th_pass:
                                    # process
                                    translation_vec_ir, scale_ind_ir, s_ir, flag_ir = tracker_ir.localize_target(
                                        scores_raw_ir_up)
                                    if tracker_ir.use_iou_net:
                                        update_scale_flag_ir = getattr(tracker_ir.params, 'update_scale_when_uncertain',
                                                                       True) or flag_ir != 'uncertain'
                                        if getattr(tracker_ir.params, 'use_classifier', True):
                                            tracker_ir.update_state(sample_pos_ir_up + translation_vec_ir)
                                        tracker_ir.refine_target_box(sample_pos_ir_up, sample_scales_ir[scale_ind_ir],
                                                                     scale_ind_ir, update_scale_flag_ir)
                                    elif getattr(tracker_ir.params, 'use_classifier', True):
                                        tracker_ir.update_state(sample_pos_ir_up + translation_vec_ir,
                                                                sample_scales_ir[scale_ind_ir])

                        else:
                            if bottom > up:
                                if bottom > th_pass:
                                    #  tHIS IS THE REAL TRAGET PROCESS THE LOCATION
                                    translation_vec_ir, scale_ind_ir, s_ir, flag_ir = tracker_ir.localize_target(
                                        scores_raw_ir_bottom)
                                    if tracker_ir.use_iou_net:
                                        update_scale_flag_ir = getattr(tracker_ir.params, 'update_scale_when_uncertain',
                                                                       True) or flag_ir != 'uncertain'
                                        if getattr(tracker_ir.params, 'use_classifier', True):
                                            tracker_ir.update_state(sample_pos_ir_bottom + translation_vec_ir)
                                        tracker_ir.refine_target_box(sample_pos_ir_bottom,
                                                                     sample_scales_ir[scale_ind_ir],
                                                                     scale_ind_ir, update_scale_flag_ir)
                                    elif getattr(tracker_ir.params, 'use_classifier', True):
                                        tracker_ir.update_state(sample_pos_ir_bottom + translation_vec_ir,
                                                                sample_scales_ir[scale_ind_ir])

                            else:
                                if up > th_pass:
                                    # process
                                    translation_vec_ir, scale_ind_ir, s_ir, flag_ir = tracker_ir.localize_target(
                                        scores_raw_ir_up)
                                    if tracker_ir.use_iou_net:
                                        update_scale_flag_ir = getattr(tracker_ir.params, 'update_scale_when_uncertain',
                                                                       True) or flag_ir != 'uncertain'
                                        if getattr(tracker_ir.params, 'use_classifier', True):
                                            tracker_ir.update_state(sample_pos_ir_up + translation_vec_ir)
                                        tracker_ir.refine_target_box(sample_pos_ir_up, sample_scales_ir[scale_ind_ir],
                                                                     scale_ind_ir, update_scale_flag_ir)
                                    elif getattr(tracker_ir.params, 'use_classifier', True):
                                        tracker_ir.update_state(sample_pos_ir_up + translation_vec_ir,
                                                                sample_scales_ir[scale_ind_ir])

                if torch.max(scores_raw_rgb[0]).item() < 0.1:
                    sample_pos_rgb_left = torch.Tensor([tracker_rgb.pos[0] - (tracker_rgb.target_sz[0])*expand_factor, tracker_rgb.pos[1]]).round()
                    test_x_rgb_left = tracker_rgb.extract_processed_sample(im_rgb, sample_pos_rgb_left, sample_scales_rgb, tracker_rgb.img_sample_sz)
                    # Compute scores
                    scores_raw_rgb_left = tracker_rgb.apply_filter(test_x_rgb_left)
                    left = torch.max(scores_raw_rgb_left[0]).item()
                    sample_pos_rgb_right = torch.Tensor([tracker_rgb.pos[0] + (tracker_rgb.target_sz[0])*expand_factor, tracker_rgb.pos[1]]).round()
                    test_x_rgb_right = tracker_rgb.extract_processed_sample(im_rgb, sample_pos_rgb_right, sample_scales_rgb, tracker_rgb.img_sample_sz)
                    # Compute scores
                    scores_raw_rgb_right = tracker_rgb.apply_filter(test_x_rgb_right)
                    right = torch.max(scores_raw_rgb_right[0]).item()
                    sample_pos_rgb_bottom = torch.Tensor([tracker_rgb.pos[0], tracker_rgb.pos[1] - (tracker_rgb.target_sz[1])*expand_factor]).round()
                    test_x_rgb_bottom = tracker_rgb.extract_processed_sample(im_rgb, sample_pos_rgb_bottom, sample_scales_rgb, tracker_rgb.img_sample_sz)
                    # Compute scores
                    scores_raw_rgb_bottom = tracker_rgb.apply_filter(test_x_rgb_bottom)
                    bottom = torch.max(scores_raw_rgb_bottom[0]).item()
                    sample_pos_rgb_up = torch.Tensor([tracker_rgb.pos[0], tracker_rgb.pos[1] + (tracker_rgb.target_sz[1])*expand_factor]).round()
                    test_x_rgb_up = tracker_rgb.extract_processed_sample(im_rgb, sample_pos_rgb_up, sample_scales_rgb, tracker_rgb.img_sample_sz)
                    # Compute scores
                    scores_raw_rgb_up = tracker_rgb.apply_filter(test_x_rgb_up)
                    up = torch.max(scores_raw_rgb_up[0]).item()

                    if left > right:
                        if left > bottom:
                            if left > up:
                                if left > th_pass:
                                    #  tHIS IS THE REAL TRAGET PROCESS THE LOCATION
                                    translation_vec_rgb, scale_ind_rgb, s_rgb, flag_rgb = tracker_rgb.localize_target(scores_raw_rgb_left)
                                    if tracker_rgb.use_iou_net:
                                        update_scale_flag_rgb = getattr(tracker_rgb.params, 'update_scale_when_uncertain',
                                                                       True) or flag_rgb != 'uncertain'
                                        if getattr(tracker_rgb.params, 'use_classifier', True):
                                            tracker_rgb.update_state(sample_pos_rgb_left + translation_vec_rgb)
                                        tracker_rgb.refine_target_box(sample_pos_rgb_left, sample_scales_rgb[scale_ind_rgb],
                                                                     scale_ind_rgb, update_scale_flag_rgb)
                                    elif getattr(tracker_rgb.params, 'use_classifier', True):
                                        tracker_rgb.update_state(sample_pos_rgb_left + translation_vec_rgb,
                                                                sample_scales_rgb[scale_ind_rgb])

                            else:
                                if up > th_pass:
                                    # process
                                    translation_vec_rgb, scale_ind_rgb, s_rgb, flag_rgb = tracker_rgb.localize_target(scores_raw_rgb_up)
                                    if tracker_rgb.use_iou_net:
                                        update_scale_flag_rgb = getattr(tracker_rgb.params, 'update_scale_when_uncertain',
                                                                       True) or flag_rgb != 'uncertain'
                                        if getattr(tracker_rgb.params, 'use_classifier', True):
                                            tracker_rgb.update_state(sample_pos_rgb_up + translation_vec_rgb)
                                        tracker_rgb.refine_target_box(sample_pos_rgb_up, sample_scales_rgb[scale_ind_rgb],
                                                                     scale_ind_rgb, update_scale_flag_rgb)
                                    elif getattr(tracker_rgb.params, 'use_classifier', True):
                                        tracker_rgb.update_state(sample_pos_rgb_up + translation_vec_rgb,
                                                                sample_scales_rgb[scale_ind_rgb])

                        else:
                            if bottom > up:
                                if bottom > th_pass:
                                    #  tHIS IS THE REAL TRAGET PROCESS THE LOCATION
                                    translation_vec_rgb, scale_ind_rgb, s_rgb, flag_rgb = tracker_rgb.localize_target(scores_raw_rgb_bottom)
                                    if tracker_rgb.use_iou_net:
                                        update_scale_flag_rgb = getattr(tracker_rgb.params, 'update_scale_when_uncertain',
                                                                       True) or flag_rgb != 'uncertain'
                                        if getattr(tracker_rgb.params, 'use_classifier', True):
                                            tracker_rgb.update_state(sample_pos_rgb_bottom + translation_vec_rgb)
                                        tracker_rgb.refine_target_box(sample_pos_rgb_bottom, sample_scales_rgb[scale_ind_rgb],
                                                                     scale_ind_rgb, update_scale_flag_rgb)
                                    elif getattr(tracker_rgb.params, 'use_classifier', True):
                                        tracker_rgb.update_state(sample_pos_rgb_bottom + translation_vec_rgb,
                                                                sample_scales_rgb[scale_ind_rgb])

                            else:
                                if up > th_pass:
                                    # process
                                    translation_vec_rgb, scale_ind_rgb, s_rgb, flag_rgb = tracker_rgb.localize_target(
                                        scores_raw_rgb_up)
                                    if tracker_rgb.use_iou_net:
                                        update_scale_flag_rgb = getattr(tracker_rgb.params, 'update_scale_when_uncertain',
                                                                       True) or flag_rgb != 'uncertain'
                                        if getattr(tracker_rgb.params, 'use_classifier', True):
                                            tracker_rgb.update_state(sample_pos_rgb_up + translation_vec_rgb)
                                        tracker_rgb.refine_target_box(sample_pos_rgb_up, sample_scales_rgb[scale_ind_rgb],
                                                                     scale_ind_rgb, update_scale_flag_rgb)
                                    elif getattr(tracker_rgb.params, 'use_classifier', True):
                                        tracker_rgb.update_state(sample_pos_rgb_up + translation_vec_rgb,
                                                                sample_scales_rgb[scale_ind_rgb])

                    else:
                        if right > bottom:
                            if right > up:
                                if right > th_pass:
                                    #  tHIS IS THE REAL TRAGET PROCESS THE LOCATION
                                    translation_vec_rgb, scale_ind_rgb, s_rgb, flag_rgb = tracker_rgb.localize_target(
                                        scores_raw_rgb_right)
                                    if tracker_rgb.use_iou_net:
                                        update_scale_flag_rgb = getattr(tracker_rgb.params, 'update_scale_when_uncertain',
                                                                       True) or flag_rgb != 'uncertain'
                                        if getattr(tracker_rgb.params, 'use_classifier', True):
                                            tracker_rgb.update_state(sample_pos_rgb_right + translation_vec_rgb)
                                        tracker_rgb.refine_target_box(sample_pos_rgb_right, sample_scales_rgb[scale_ind_rgb],
                                                                     scale_ind_rgb, update_scale_flag_rgb)
                                    elif getattr(tracker_rgb.params, 'use_classifier', True):
                                        tracker_rgb.update_state(sample_pos_rgb_right + translation_vec_rgb,
                                                                sample_scales_rgb[scale_ind_rgb])

                            else:
                                if up > th_pass:
                                    # process
                                    translation_vec_rgb, scale_ind_rgb, s_rgb, flag_rgb = tracker_rgb.localize_target(
                                        scores_raw_rgb_up)
                                    if tracker_rgb.use_iou_net:
                                        update_scale_flag_rgb = getattr(tracker_rgb.params, 'update_scale_when_uncertain',
                                                                       True) or flag_rgb != 'uncertain'
                                        if getattr(tracker_rgb.params, 'use_classifier', True):
                                            tracker_rgb.update_state(sample_pos_rgb_up + translation_vec_rgb)
                                        tracker_rgb.refine_target_box(sample_pos_rgb_up, sample_scales_rgb[scale_ind_rgb],
                                                                     scale_ind_rgb, update_scale_flag_rgb)
                                    elif getattr(tracker_rgb.params, 'use_classifier', True):
                                        tracker_rgb.update_state(sample_pos_rgb_up + translation_vec_rgb,
                                                                sample_scales_rgb[scale_ind_rgb])

                        else:
                            if bottom > up:
                                if bottom > th_pass:
                                    #  tHIS IS THE REAL TRAGET PROCESS THE LOCATION
                                    translation_vec_rgb, scale_ind_rgb, s_rgb, flag_rgb = tracker_rgb.localize_target(
                                        scores_raw_rgb_bottom)
                                    if tracker_rgb.use_iou_net:
                                        update_scale_flag_rgb = getattr(tracker_rgb.params, 'update_scale_when_uncertain',
                                                                       True) or flag_rgb != 'uncertain'
                                        if getattr(tracker_rgb.params, 'use_classifier', True):
                                            tracker_rgb.update_state(sample_pos_rgb_bottom + translation_vec_rgb)
                                        tracker_rgb.refine_target_box(sample_pos_rgb_bottom,
                                                                     sample_scales_rgb[scale_ind_rgb],
                                                                     scale_ind_rgb, update_scale_flag_rgb)
                                    elif getattr(tracker_rgb.params, 'use_classifier', True):
                                        tracker_rgb.update_state(sample_pos_rgb_bottom + translation_vec_rgb,
                                                                sample_scales_rgb[scale_ind_rgb])
                            else:
                                if up > th_pass:
                                    # process
                                    translation_vec_rgb, scale_ind_rgb, s_rgb, flag_rgb = tracker_rgb.localize_target(
                                        scores_raw_rgb_up)
                                    if tracker_rgb.use_iou_net:
                                        update_scale_flag_rgb = getattr(tracker_rgb.params, 'update_scale_when_uncertain',
                                                                       True) or flag_rgb != 'uncertain'
                                        if getattr(tracker_rgb.params, 'use_classifier', True):
                                            tracker_rgb.update_state(sample_pos_rgb_up + translation_vec_rgb)
                                        tracker_rgb.refine_target_box(sample_pos_rgb_up, sample_scales_rgb[scale_ind_rgb],
                                                                     scale_ind_rgb, update_scale_flag_rgb)
                                    elif getattr(tracker_rgb.params, 'use_classifier', True):
                                        tracker_rgb.update_state(sample_pos_rgb_up + translation_vec_rgb,
                                                                sample_scales_rgb[scale_ind_rgb])

            if visulization:
                _gt = label_res_IR['gt_rect'][frame_id]
                _exist = label_res_IR['exist'][frame_id]
                if _exist:
                    cv2.rectangle(frame_IR, (int(_gt[0]), int(_gt[1])), (int(_gt[0] + _gt[2]), int(_gt[1] + _gt[3])),
                                  (0, 255, 0))
                cv2.putText(frame_IR, 'exist' if _exist else 'not exist',
                            (frame_IR.shape[1] // 2 - 20, 30), 1, 2, (0, 255, 0) if _exist else (0, 0, 255), 2)

                cv2.putText(frame_IR, 'Max score IR = {:.2f}'.format(max_score_ir),
                            (frame_IR.shape[1] // 2 - 60, 50), 1, 2, (255, 0, 0), 2)

                cv2.putText(frame_IR, 'Max score RGB = {:.2f}'.format(max_score_rgb),
                            (frame_IR.shape[1] // 2 - 60, 70), 1, 2, (255, 0, 0), 2)

                cv2.rectangle(frame_IR, (int(new_state[0]), int(new_state[1])), (int(new_state[0] + new_state[2]), int(new_state[1] + new_state[3])),
                              (0, 255, 255))
                # cv2.imshow(video_name, frame_IR)
                # write the frame
                out.write(frame_IR)

                cv2.waitKey(1)

            frame_id += 1
        if visulization:
            # cv2.destroyAllWindows()
            out.release()
        # save result
        output_file = os.path.join(output_dir, '%s_%s.txt' % (video_name, mode))
        with open(output_file, 'w') as f:
            json.dump({'res': out_res}, f)
        # output_conf_file = os.path.join(output_video_path, '%s_%s_conf_IR.txt' % (video_name, mode))
        # with open(output_conf_file, 'w') as f:
        #     json.dump(max_score_ir, f)

        mixed_measure = eval(out_res, label_res_IR)
        overall_performance.append(mixed_measure)
        print('[%03d/%03d] %20s %5s Fixed Measure: %.03f' % (video_id, video_num, video_name, mode, mixed_measure))

    print('[Overall] %5s Mixed Measure: %.03f\n' % (mode, np.mean(overall_performance)))


if __name__ == '__main__':
    main(mode='IR', visulization=True)
