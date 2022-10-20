from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone
import random

def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.use_gpu = True

    params.image_sample_size = 18*16
    params.search_area_scale = 5.6  # change

    # add
    # params.LOST_INSTANCE_SIZE = 831
    params.scale_factors = [1.30, 1.15, 1, 0.75, 0.5]   # [1.30, 1.15, 1, 0.75, 0.5] 0.684
    params.scale_factors1 = [random.uniform(1.30, 1.28), 1.15, 1, 0.75, 0.5]   # [1.30, 1.15, 1, 0.75, 0.5] 0.684
    params.scale_factors2 = [1.15, 1, 0.75, 0.5]   # [1.25, 1.15, 1, 0.75, 0.5] 0.680
    params.CONFIDENCE_LOW = 0.25   # 0.25best
    params.CONFIDENCE_HIGH = 1.02  # 0.99 best
    # params.use_iounet_pos_for_learning = True

    # Learning parameters
    params.sample_memory_size = 50
    params.learning_rate = 0.01
    params.init_samples_minimum_weight = 0.25
    params.train_skipping = 28  # change from 25

    # Net optimization params
    params.update_classifier = True
    params.net_opt_iter = 10
    # params.net_opt_iter = 15
    params.net_opt_update_iter = 2
    params.net_opt_hn_iter = 1
    # add no use
    # params.low_score_opt_threshold = 0.37
    # params.net_opt_low_iter = 1

    # Detection parameters
    params.window_output = False

    # Init augmentation parameters
    params.use_augmentation = True
    params.augmentation = {'fliplr': True,
                           'rotate': [10, -10, 45, -45],
                           'blur': [(3,1), (1, 3), (2, 2)],
                           'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6, -0.6)],
                           # 'relativeshift': [(0.75, 0.75), (-0.75, 0.75), (0.75, -0.75), (-0.75, -0.75)],
                           'dropout': (2, 0.2)}

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1/3
    # Advanced localization parameters
    params.advanced_localization = True
    params.target_not_found_threshold = 0.20  # 0.20 best
    params.distractor_threshold = 0.8
    params.hard_negative_threshold = 0.5
    params.target_neighborhood_scale = 2.2
    # params.target_neighborhood_scale = 2.3
    params.dispalcement_scale = 0.8
    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = True

    # IoUnet parameters
    params.iounet_augmentation = False
    params.iounet_use_log_scale = True
    params.iounet_k = 3
    params.num_init_random_boxes = 9
    params.box_jitter_pos = 0.1
    params.box_jitter_sz = 0.5
    params.maximal_aspect_ratio = 8
    params.box_refinement_iter = 5
    params.box_refinement_step_length = 1
    params.box_refinement_step_decay = 1

    # params.net = NetWithBackbone(net_path='dimp50.pth',
    params.net = NetWithBackbone(net_path='DiMPnet_ep0060_best.pth',
                                 use_gpu=params.use_gpu)

    params.vot_anno_conversion_type = 'preserve_area'

    return params
