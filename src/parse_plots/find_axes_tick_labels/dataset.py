import json
import os
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import torch.utils.data
from torch.utils.data.dataset import T_co
from torchvision import datapoints, io
from torchvision.transforms.v2 import functional as F


class PlotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        plot_ids: List[str],
        plots_dir,
        annotations_dir,
        transform
    ):
        super().__init__()
        plot_ids = set(plot_ids)
        plot_files = os.listdir(plots_dir)
        self._plot_files = [x for x in plot_files if Path(x).stem in plot_ids]
        self._plots_dir = Path(plots_dir)
        self._annotations_dir = Path(annotations_dir)
        self._transform = transform

    def __getitem__(self, index) -> T_co:
        id = Path(self._plot_files[index]).stem
        img = io.read_image(str(self._plots_dir / f'{id}.jpg'))
        img = datapoints.Image(img)

        with open(self._annotations_dir / f'{id}.json') as f:
            a = json.load(f)

        tick_labels = [x for x in a['text'] if x['role'] == 'tick_label']
        boxes = self._get_bboxes(
            tick_labels=tick_labels,
            img=img
        )
        labels = self._get_labels(axes=a['axes'], tick_labels=tick_labels)
        masks = self._get_masks(
            tick_labels=tick_labels,
            img=img
        )
        if self._transform is not None:
            img, boxes, masks, labels = self._transform(
                img, boxes, masks, labels)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks
        }

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
        """Gets whether axis label is part of x or y axis"""
        axes_label_map = {
            'x-axis': 0,
            'y-axis': 1
        }

        tick_axis_map = {}
        for axis in axes_label_map:
            ticks = axes[axis]['ticks']
            for tick in ticks:
                tick_axis_map[tick['id']] = axes_label_map[axis]

        labels = torch.tensor(
            [tick_axis_map[tick['id']] for tick in tick_labels],
            dtype=torch.int64
        )
        return labels

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
