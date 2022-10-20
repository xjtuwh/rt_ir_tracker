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

from siamfc import TrackerSiamFC


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


def main(mode='IR', visulization=False):
    assert mode in ['IR', 'RGB'], 'Only Support IR or RGB to evalute'

    # setup experiments
    video_paths = glob.glob(os.path.join('/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/anti_UAV/dataset', 'hard', '*'))    # data path
    video_num = len(video_paths)
    tracker1 = 'dimp_det3_challenge'
    # tracker2 = 'dimp_det_challenge'
    tracker3 = 'dimp_det2_challenge'
    output_dir1 = os.path.join('results', tracker1)
    # output_dir2 = os.path.join('results', tracker2)
    output_dir3 = os.path.join('results', tracker3)
    output_video_path = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/WH/wh2020/anti-UAV/visualize/' + 'sum1/'
    if not os.path.exists(output_video_path):
        os.makedirs(output_video_path)

    # run tracking experiments and report performance
    for video_id, video_path in enumerate(video_paths, start=1):
        video_name = os.path.basename(video_path)
        video_file = os.path.join(video_path, '%s.mp4'%mode)
        res_file = os.path.join(video_path, '%s_label.json'%mode)
        with open(res_file, 'r') as f:
            label_res = json.load(f)

        output_file1 = os.path.join(output_dir1, '%s_%s.txt' % (video_name, mode))
        with open(output_file1, 'r') as f:
            siamfc_res = json.load(f)

        # output_file2 = os.path.join(output_dir2, '%s_%s.txt' % (video_name, mode))
        # with open(output_file2, 'r') as f:
        #     atom_res = json.load(f)

        output_file3 = os.path.join(output_dir3, '%s_%s.txt' % (video_name, mode))
        with open(output_file3, 'r') as f:
            our_res = json.load(f)

        init_rect = label_res['gt_rect'][0]
        capture = cv2.VideoCapture(video_file)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path + video_name + '.mp4', fourcc, 20.0, (640, 512))

        frame_id = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                capture.release()
                break

            if visulization:
                # _gt = label_res['gt_rect'][frame_id]
                # _exist = label_res['exist'][frame_id]
                _siamfc = siamfc_res['res'][frame_id]
                # _atom = atom_res['res'][frame_id]
                _ours = our_res['res'][frame_id]
                # if _exist:
                #     cv2.rectangle(frame, (int(_gt[0]), int(_gt[1])), (int(_gt[0] + _gt[2]), int(_gt[1] + _gt[3])),
                #                   (0, 255, 0))
                # cv2.putText(frame, 'exist' if _exist else 'not exist',
                #             (frame.shape[1] // 2 - 20, 30), 1, 2, (0, 255, 0) if _exist else (0, 0, 255), 2)
                cv2.putText(frame, '#{:d}'.format(frame_id),
                            (frame.shape[1] - 100, 70), 1, 2, (0, 255, 0), 2)

                if len(_siamfc) > 1:
                    cv2.rectangle(frame, (int(_siamfc[0]), int(_siamfc[1])), (int(_siamfc[0] + _siamfc[2]), int(_siamfc[1] + _siamfc[3])),
                                  (0, 255, 255))
                # if len(_atom) > 1:
                #     cv2.rectangle(frame, (int(_atom[0]), int(_atom[1])), (int(_atom[0] + _atom[2]), int(_atom[1] + _atom[3])),
                #                   (255, 0, 0))
                if len(_ours) > 1:
                    cv2.rectangle(frame, (int(_ours[0]), int(_ours[1])), (int(_ours[0] + _ours[2]), int(_ours[1] + _ours[3])),
                                  (0, 0, 255))
                # cv2.imshow(video_name, frame)
                # cv2.waitKey(1)
                out.write(frame)

            frame_id += 1
        if visulization:
            # cv2.destroyAllWindows()
            out.release()

if __name__ == '__main__':
    main(mode='IR', visulization=True)
