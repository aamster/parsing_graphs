import re
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes


def draw_bounding_box_and_seg_mask(img: torch.tensor, targets: Dict):
    colors = (
        [(0, 255, 0) if targets['labels'][i].item() == 0
         else (0, 255, 255)
         for i in range(len(targets['labels']))]
    )
    img = (img * 255).type(torch.uint8)
    img = draw_bounding_boxes(
        image=img,
        boxes=targets['boxes'],
        colors=colors
    )

    img = draw_segmentation_masks(
        image=img,
        masks=targets['masks'].type(torch.bool),
        colors=colors
    )
    plt.imshow(img.moveaxis(0, 2))
    plt.show()


def string_to_float(x: str) -> float:
    x = re.sub(r'[ ,%$]', '', x)
    dot_count = len([c for c in x if c == '.'])
    if dot_count > 1:
        # it's european style like 10.000.000?
        x = x.replace('.', '')
    x = float(x)
    return x


def threshold_soft_masks(preds: List[Dict]):
    """

    :param preds:
    :return:
        `preds` with `masks` thresholded inplace
    """
    for pred_idx, pred in enumerate(preds):
        masks = torch.zeros_like(preds[pred_idx]['masks'],
                                 dtype=torch.uint8)
        for mask_idx, mask in enumerate(pred['masks']):
            mask = (mask > 0.5).type(torch.uint8)
            masks[mask_idx] = mask
        preds[pred_idx]['masks'] = masks
        if len(preds[pred_idx]['masks'].shape) == 4:
            preds[pred_idx]['masks'] = preds[pred_idx]['masks'].squeeze(
                dim=1)
    return preds


def convert_masks_to_tensor(target):
    for i in range(len(target)):
        if type(target[i]['masks']) is not torch.Tensor:
            target[i]['masks'] = target[i]['masks'].data
    return target
