"""
baseline for Anti-UAV
https://anti-uav.github.io/
"""
from __future__ import absolute_import
import os
import glob
import json
import numpy as np



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


def main(mode='IR', visulization=False):
    assert mode in ['IR', 'RGB'], 'Only Support IR or RGB to evalute'

    # setup experiments
    video_paths = glob.glob(os.path.join('/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/anti_UAV/dataset', 'test-dev', '*'))    # data path
    video_num = len(video_paths)
    tracker1 = 'dimp_det2'

    output_dir1 = os.path.join('results', tracker1)
    overall_performance = []
    # run tracking experiments and report performance
    for video_id, video_path in enumerate(video_paths, start=1):
        video_name = os.path.basename(video_path)
        
        res_file = os.path.join(video_path, '%s_label.json'%mode)
        with open(res_file, 'r') as f:
            label_res = json.load(f)

        output_file1 = os.path.join(output_dir1, '%s_%s.txt' % (video_name, mode))
        with open(output_file1, 'r') as f:
            our_res = json.load(f)

        mixed_measure = eval(our_res['res'], label_res)
        overall_performance.append(mixed_measure)
        print('[%03d/%03d] %20s %5s Fixed Measure: %.03f' % (video_id, video_num, video_name, mode, mixed_measure))
    print('[Overall] %5s Mixed Measure: %.03f\n' % (mode, np.mean(overall_performance)))



if __name__ == '__main__':
    main(mode='IR', visulization=False)
