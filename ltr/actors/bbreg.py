from . import BaseActor
import torch
import math


def bbox_overlaps_ciou(bboxes1, bboxes2, iou):
    w1 = bboxes1[2]
    h1 = bboxes1[3]
    w2 = bboxes2[2]
    h2 = bboxes2[3]

    center_x1 = bboxes1[2] + bboxes1[0] / 2
    center_y1 = bboxes1[3] + bboxes1[1] / 2
    center_x2 = bboxes2[2] + bboxes2[0] / 2
    center_y2 = bboxes2[3] + bboxes2[1] / 2

    out_max_xy = torch.max(bboxes1[2:],bboxes2[2:])
    out_min_xy = torch.min(bboxes1[:2],bboxes2[:2])

    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[0] ** 2) + (outer[1] ** 2)
    u = (inter_diag) / outer_diag

    # with torch.no_grad():
    #     arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
    #     v = (4 / (math.pi ** 2)) * torch.pow(arctan, 2)
    #     S = iou
    #     alpha = v / (S + v)
    # cious = u + alpha * v
    cious = u
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    return cious


class AtomActor(BaseActor):
    """ Actor for training the IoU-Net in ATOM"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals' and 'proposal_iou'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        iou_pred = self.net(data['train_images'], data['test_images'], data['train_anno'], data['test_proposals'])

        iou_pred = iou_pred.view(-1, iou_pred.shape[2])
        iou_gt = data['proposal_iou'].view(-1, data['proposal_iou'].shape[2])

        loss = self.objective(iou_pred, iou_gt)

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss.item()}

        return loss, stats