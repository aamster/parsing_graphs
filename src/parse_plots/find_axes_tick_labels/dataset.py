import json
import os
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch.utils.data
import torchvision
from torch.utils.data.dataset import T_co

torchvision.disable_beta_transforms_warning()

from torchvision import datapoints, io
from torchvision.transforms.v2 import functional as F


# 0 indicates background - IMPORTANT !!
axes_label_map = {
    'x-axis': 1,
    'y-axis': 2
}


class FindAxesTickLabelsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        plot_ids: List[str],
        plots_dir,
        transform,
        annotations_dir: Optional[Path] = None,
        return_axes_tick_text: bool = False,
        is_train: bool = True
    ):
        super().__init__()
        # these lack labeled ticks
        bad_ids = ['733b9b19e09a', 'aa9df520a5f2', '04296b42ba61',
                   '3968efe9cbfc', '6ce4bc728dd5', 'd0cf883b1e13']

        plot_ids = [x for x in plot_ids if x not in bad_ids]
        plot_files = os.listdir(plots_dir)
        self._plot_files = [x for x in plot_files if Path(x).stem in plot_ids]
        self._plots_dir = Path(plots_dir)
        self._annotations_dir = Path(annotations_dir) \
            if annotations_dir is not None else None
        self._transform = transform
        self._return_axes_tick_text = return_axes_tick_text
        self._is_train = is_train

    def __getitem__(self, index) -> T_co:
        id = Path(self._plot_files[index]).stem
        img = io.read_image(str(self._plots_dir / f'{id}.jpg'))
        img = datapoints.Image(img)

        if not self._is_train:
            if self._transform is not None:
                img = self._transform(img)
            return img, {'image_id': id}

        with open(self._annotations_dir / f'{id}.json') as f:
            a = json.load(f)

        tick_labels = [x for x in a['text'] if x['role'] == 'tick_label']
        labels, tick_labels = \
            self._get_labels(axes=a['axes'], tick_labels=tick_labels)

        boxes = self._get_bboxes(
            tick_labels=tick_labels,
            img=img
        )
        masks = self._get_masks(
            tick_labels=tick_labels,
            img=img
        )
        if self._transform is not None:
            img, boxes, masks, labels = self._transform(
                img, boxes, masks, labels)

        target = {
            'image_id': id,
            'boxes': boxes,
            'labels': labels,
            'masks': masks
        }
        if self._return_axes_tick_text:
            target['axes_tick_text'] = [x['text'] for x in tick_labels]

        return img, target

    def __len__(self):
        return len(self._plot_files)

    @staticmethod
    def _get_bboxes(
        tick_labels: List[Dict],
        img: datapoints.Image
    ):
        """Gets bounding boxes around tick label text"""
        n_points = 4
        bboxes = []
        for i, text in enumerate(tick_labels):
            polygon = text['polygon']
            polygon = [[[polygon[f'x{i}'], polygon[f'y{i}']]] for i in
                       range(n_points)]
            polygon = np.array(polygon, dtype='float32')
            bbox = cv2.boundingRect(polygon)
            bboxes.append(
                [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        bboxes = datapoints.BoundingBox(
            bboxes,
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=F.get_spatial_size(img),
            dtype=torch.float
        )

        return bboxes

    @staticmethod
    def _get_labels(axes: Dict, tick_labels: List[Dict]):
        """Gets whether axis label is part of x or y axis
        Also modifies tick_labels to exclude any not in axes
        """
        tick_axis_map = {}
        for axis in axes_label_map:
            ticks = axes[axis]['ticks']
            for tick in ticks:
                tick_axis_map[tick['id']] = axes_label_map[axis]

        # There are some labels erroneously marked as tick labels that are
        # not tick labels and don't appear in "ticks"
        tick_labels = [x for x in tick_labels if x['id'] in tick_axis_map]

        labels = torch.tensor(
            [tick_axis_map[tick['id']] for tick in tick_labels],
            dtype=torch.int64
        )
        return labels, tick_labels

    @staticmethod
    def _get_masks(
        tick_labels: List[Dict],
        img: datapoints.Image
    ):
        masks = np.zeros((len(tick_labels), img.shape[1], img.shape[2]))
        n_points = 4
        for i, text in enumerate(tick_labels):
            polygon = text['polygon']
            polygon = np.array(
                [[polygon[f'x{i}'], polygon[f'y{i}']] for i in
                 range(n_points)])
            mask = np.zeros((img.shape[1], img.shape[2]), dtype='uint8')
            cv2.fillPoly(mask, pts=[polygon], color=1)
            masks[i] = mask
        masks = datapoints.Mask(masks, dtype=torch.uint8)
        return masks
