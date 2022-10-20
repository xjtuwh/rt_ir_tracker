import torch
import os
import os.path
import numpy as np
import pandas
from collections import OrderedDict

from ltr.data.image_loader import default_image_loader
from .base_dataset import BaseDataset
from ltr.admin.environment import env_settings

def list_sequences(root):

    sequence_list = []

    anno_dir = os.path.join(root, "annotations")
    seq_dir = os.path.join(root, "sequences")
    for filename in os.listdir(seq_dir):
        sequence_list.append(filename)

    return sequence_list


class visdrone(BaseDataset):

    def __init__(self, root=None, image_loader=default_image_loader, set_ids=None):

        root = env_settings().visdrone_train_dir if root is None else root
        super().__init__(root, image_loader)
        self.sequence_list = list_sequences(self.root)

    def get_name(self):
        return 'visdrone'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_anno(self, anno_path):
        gt = pandas.read_csv(anno_path, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        seq_path = os.path.join(self.root, 'sequences', seq_name)
        anno_path = os.path.join(self.root, 'annotations', seq_name+'.txt')
        return seq_path, anno_path

    def get_sequence_info(self, seq_id):
        seq_path, anno_path = self._get_sequence_path(seq_id)
        anno = self._read_anno(anno_path)
        target_visible = (anno[:,2]>0) & (anno[:,3]>0)
        visible = target_visible.byte()
        return {'bbox': anno, 'valid': target_visible, 'visible': visible}

    def _get_frame_path(self,seq_path,frame_id):
        return os.path.join(seq_path, 'img{:07}.jpg'.format(frame_id+1))

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path, anno_path = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame(seq_path, f) for f in frame_ids]

        # if anno is None:
        #     anno = self._read_anno(anno_path)
        #
        # # Return as list of tensors
        # anno_frames = [anno[f_id, :] for f_id in frame_ids]

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