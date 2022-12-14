import torch.nn as nn
import torch.optim as optim
import torchvision.transforms

from ltr.dataset import Lasot, Got10k, TrackingNet, MSCOCOSeq, VOTIR_TRAIN, VOTIR_VAL, IR234, UAV, GDHW
from ltr.data import processing, sampler, LTRLoader
import ltr.models.bbreg.atom as atom_models
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as dltransforms


def run(settings):
    # Most common settings are assigned in the settings struct
    settings.description = 'ATOM IoUNet with default settings.'
    settings.print_interval = 1                                 # How often to print loss and other info
    settings.batch_size = 2                                     # Batch size 64
    settings.num_workers = 8                                    # Number of workers for image loading 4
    # settings.normalize_mean = [0.393, 0.393, 0.393]             # Normalize mean (default pytorch ImageNet values)[0.485, 0.456, 0.406]
    # settings.normalize_std = [0.123, 0.123, 0.123]              # Normalize std (default pytorch ImageNet values)[0.229, 0.224, 0.225]
    settings.normalize_mean = [0.320, 0.328, 0.309]             # IR BGR
    settings.normalize_std = [0.198, 0.198, 0.198]

    settings.search_area_factor = 5.0                           # Image patch size relative to target size
    settings.feature_sz = 18                                    # Size of feature map
    settings.output_sz = settings.feature_sz * 16               # Size of input image patches
    # Settings for the image sample and proposal generation
    settings.center_jitter_factor = {'train': 0, 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0, 'test': 0.5}
    settings.proposal_params = {'min_iou': 0.01, 'boxes_per_frame': 16, 'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]}

    # Train datasets
    # lasot_train = Lasot(settings.env.lasot_dir, split='train')
    # got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
    # trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(11)))
    # coco_train = MSCOCOSeq(settings.env.coco_dir)
    # Validation datasets
    # got10k_val = Got10k(settings.env.got10k_dir, split='votval')
    # ir_train = IR234()
    # vot_ir_train = VOTIR_TRAIN()
    ir_val = VOTIR_VAL()
    # uav_train = UAV()
    gdhw = GDHW()

    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = dltransforms.ToGrayscale(probability=1)

    # The augmentation transform applied to the training set (individually to each image in the pair)
    # transform_train = torchvision.transforms.Compose([dltransforms.ToTensorAndJitter(0.2),
    #                                                   torchvision.transforms.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)])
    transform_train = torchvision.transforms.Compose([dltransforms.ToTensorAndJitter(0.2),
                                                      dltransforms.RandomHorizontalFlip(),
                                                      dltransforms.Blur((3, 1)),
                                                      dltransforms.GridMask(),
                                                      torchvision.transforms.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)])

    # The augmentation transform applied to the validation set (individually to each image in the pair)
    transform_val = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)])

    # Data processing to do on the training pairs
    data_processing_train = processing.ATOMProcessing(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      proposal_params=settings.proposal_params,
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)

    # Data processing to do on the validation pairs
    data_processing_val = processing.ATOMProcessing(search_area_factor=settings.search_area_factor,
                                                    output_sz=settings.output_sz,
                                                    center_jitter_factor=settings.center_jitter_factor,
                                                    scale_jitter_factor=settings.scale_jitter_factor,
                                                    mode='sequence',
                                                    proposal_params=settings.proposal_params,
                                                    transform=transform_val,
                                                    joint_transform=transform_joint)

    # The sampler for training
    dataset_train = sampler.ATOMSampler([gdhw], [1],
                                        samples_per_epoch=1000*settings.batch_size, max_gap=50,
                                        processing=data_processing_train)
    # dataset_train = sampler.ATOMSampler([lasot_train, got10k_train, trackingnet_train, coco_train, ir_train, uav_train, vot_ir_train, gdhw], [1, 1, 1, 1, 1, 1, 1, 1],
    #                                     samples_per_epoch=1000*settings.batch_size, max_gap=50,
    #                                     processing=data_processing_train)
    # dataset_train = sampler.ATOMSampler([got10k_train, ir_train, uav_train, vot_ir_train, gdhw], [1, 1, 1, 1, 1],
    #                                     samples_per_epoch=1000*settings.batch_size, max_gap=50,
    #                                     processing=data_processing_train)

    # The loader for training
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    # The sampler for validation  trackingnet_val
    dataset_val = sampler.ATOMSampler([ir_val], [1], samples_per_epoch=500*settings.batch_size, max_gap=50,
                                      processing=data_processing_val)

    # The loader for validation
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1)

    # Create network
    net = atom_models.atom_resnet18(backbone_pretrained=True)
    # net = atom_models.atom_resnet50(backbone_pretrained=True)
    # net = atom_models.atom_aog(backbone_pretrained=True)

    # Set objective
    objective = nn.MSELoss()

    # Create actor, which wraps network and objective
    actor = actors.AtomActor(net=net, objective=objective)

    # Optimizer
    optimizer = optim.Adam(actor.net.bb_regressor.parameters(), lr=1e-3)  # , weight_decay=0.00002

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    # Create trainer
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(80, load_latest=True, fail_safe=False)
