from typing import Dict

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
