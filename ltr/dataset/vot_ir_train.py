import torch
import os
import os.path
import numpy as np
import pandas
import random
from collections import OrderedDict

from ltr.data.image_loader import default_image_loader
from .base_dataset import BaseDataset
from ltr.admin.environment import env_settings


def list_sequences(root):
    sequence_name = ['bird',
                     'birds',
                     'boat1',
                     'boat2',
                     'car1',
                     'car2',
                     'depthwise_crossing',
                     'dog',
                     'mixed_distractors',
                     'quadrocopter',
                     'quadrocopter2',
                     'ragged',
                     'saturated',
                     'selma',
                     'soccer']
    sequence_list = []
    for filename in sequence_name:
        sequence_list.append(filename)

    return sequence_list


class VOTIR_TRAIN(BaseDataset):
    """VOT_IR2018 dataset

    Publication:
        The sixth Visual Object Tracking VOT2018 challenge results.
        Matej Kristan, Ales Leonardis, Jiri Matas, Michael Felsberg, Roman Pfugfelder, Luka Cehovin Zajc, Tomas Vojir,
        Goutam Bhat, Alan Lukezic et al.
        ECCV, 2018
        https://prints.vicos.si/publications/365

    Download the dataset from http://www.votchallenge.net/vot2018/dataset.html"""
    def __init__(self, root=None, image_loader=default_image_loader):
        root = env_settings().vot_ir_train_dir if root is None else root
        super().__init__(root, image_loader)
        self.sequence_list = list_sequences(self.root)

    def get_name(self):
        return 'vot_ir'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_anno(self, anno_path):
        try:
            # ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float32)
            ground_truth_rect = pandas.read_csv(anno_path, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                                 low_memory=False).values
        except:
            # ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)
            ground_truth_rect = pandas.read_csv(anno_path, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                                 low_memory=False).values

        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)
        return torch.tensor(ground_truth_rect)

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        seq_path = os.path.join(self.root, seq_name)
        anno_path = os.path.join(self.root, seq_name, 'groundtruth.txt')
        return seq_path, anno_path

    def get_sequence_info(self, seq_id):
        seq_path, anno_path = self._get_sequence_path(seq_id)
        anno = self._read_anno(anno_path)
        target_visible = (anno[:, 2] > 0) & (anno[:, 3] > 0)
        visible = target_visible.byte()
        return {'bbox': anno, 'valid': target_visible, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:08}.png'.format(frame_id + 1))

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path, anno_path = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame(seq_path, f) for f in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # Create anno dict
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta