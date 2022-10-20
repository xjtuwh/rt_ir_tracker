class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/WH/wh2020/anti-UAV/Anti-UAV-master/Eval/ltr/save_models'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/LaSOT/LaSOTBenchmark'
        self.got10k_dir = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/got_10k/train'
        self.trackingnet_dir = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/TrackingNet'
        self.coco_dir = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/coco'
        self.imagenet_dir = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/ILSVRC2012_img_train'
        self.imagenetdet_dir = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/ILSVRC2017/ILSVRC    '
        self.visdrone_train_dir = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/VisDrone2019/visdrone2019_datasets/VisDrone2018-SOT-train'
        self.visdrone_val_dir = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/VisDrone2019/visdrone2019_datasets/VisDrone2018-SOT-val'
        self.visdrone_val_dir = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/VisDrone2019/visdrone2019_datasets/VisDrone2018-SOT-val'
        self.vot_ir_train_dir = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/WH/wh2019/VOT2019/vot-toolkit/vot_ir/sequences'
        self.vot_ir_val_dir = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/WH/wh2019/VOT2019/vot-toolkit/vot_ir/sequences1'
        self.ir234_dir = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/RGB-T234'
        self.uav_dir = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/anti_UAV/dataset/test-dev'
        self.gdhw_dir = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/DataSet/gdhw_ir/dataset'