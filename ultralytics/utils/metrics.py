def shape_iou(box1, box2, xywh=True, scale=0, eps=1e-7):
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
 
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
 
    union = w1 * h1 + w2 * h2 - inter + eps
 
    iou = inter / union
  
    ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
    hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps                            # convex diagonal squared
    center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
    center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    center_distance = hh * center_distance_x + ww * center_distance_y
    distance = center_distance / c2
  
    omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
    omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
    shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
    
    iou = iou - distance - 0.5 * ( shape_cost)
    return iou  # IoU

def wasserstein_loss(pred, target, eps=1e-7, constant=12.8):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.
    Code is modified from https://github.com/Zzh-tju/CIoU.
    Args:
        pred (Tensor): Predicted bboxes of format (x_center, y_center, w, h),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """

    center1 = pred[:, :2]
    center2 = target[:, :2]

    whs = center1[:, :2] - center2[:, :2]

    center_distance = whs[:, 0] * whs[:, 0] + whs[:, 1] * whs[:, 1] + eps #

    w1 = pred[:, 2]  + eps
    h1 = pred[:, 3]  + eps
    w2 = target[:, 2] + eps
    h2 = target[:, 3] + eps

    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wasserstein_2 = center_distance + wh_distance
    return torch.exp(-torch.sqrt(wasserstein_2) / constant)
