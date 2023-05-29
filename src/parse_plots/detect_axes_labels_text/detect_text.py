import re
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pytesseract
import torch
from PIL import Image
from torchvision import datapoints
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

from parse_plots.find_axes_tick_labels.dataset import axes_label_map


class DetectText:
    def __init__(
        self,
        axes_segmentations: Dict,
        images_dir: Path,
        plot_types: Dict[str, str],
        segmentation_resize=(448, 448)
    ):
        self._axes_segmentations = axes_segmentations
        self._images_dir = images_dir
        self._segmentation_resize = segmentation_resize
        self._plot_types = plot_types

    def run(self):
        res = {}
        for file_id, pred in tqdm(
                self._axes_segmentations.items(),
                total=len(self._axes_segmentations)):
            img = Image.open(f'{self._images_dir / file_id}.jpg')
            img = datapoints.Image(img)

            boxes = pred['boxes']
            masks = pred['masks']
            labels = pred['labels']
            boxes, masks = self._resize_boxes_and_masks(
                img=img,
                boxes=boxes,
                masks=masks)

            res[file_id] = {
                axis: self._get_labels_for_axis(
                    axis=axis,
                    labels=labels,
                    boxes=boxes,
                    masks=masks,
                    img=img,
                    plot_type=self._plot_types[file_id]
                ) for axis in ('x-axis', 'y-axis')}
        return res

    def _get_labels_for_axis(
        self,
        img,
        axis,
        labels,
        boxes,
        masks,
        plot_type
    ):
        def sort_boxes(boxes):
            box_sort_vals = torch.tensor(
                [box[0].item() if axis == 'x-axis' else -box[3].item() for box
                 in boxes])
            sorted_idx = torch.argsort(box_sort_vals)
            return sorted_idx

        axis_labels = torch.where(labels == axes_label_map[axis])[0]
        axis_boxes = boxes[axis_labels]

        sort_idx = sort_boxes(boxes=axis_boxes)
        axis_boxes = axis_boxes[sort_idx]
        axis_masks = masks[axis_labels][sort_idx]

        axis_text = []
        for box_idx in range(len(axis_boxes)):
            text = self._get_text(
                img=img,
                mask=axis_masks[box_idx],
                axis_name=axis,
                plot_type=plot_type
            )
            axis_text.append(text)
        return axis_text

    def _resize_boxes_and_masks(self, img, boxes, masks):
        boxes = datapoints.BoundingBox(
            boxes,
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=self._segmentation_resize,
            dtype=torch.float
        )
        masks = datapoints.Mask(masks, dtype=torch.uint8)

        boxes, masks = transforms.Resize(size=img.shape[1:])(boxes, masks)
        return boxes, masks

    def _get_text(
        self,
        img: torch.tensor,
        mask: torch.tensor,
        axis_name: str,
        plot_type: str
    ):
        img = self._rotate_cropped_text(img=img, mask=mask)

        def invert_background(img):
            """tesseract does poorly when bg is not white.
            inverts img so that the background is white"""
            return TF.invert(Image.fromarray(img))

        # found by taking small sample of imgs with white bg
        white_bg_mean = 225.19422039776728
        white_bg_std = 4.534304556306066

        z_score = (img.mean() - white_bg_mean) / white_bg_std
        if abs(z_score) > 6:
            #         print(img.mean())
            #         print('inverting')
            #         print('before...')
            #         plt.imshow(img)
            #         plt.show()
            img = invert_background(img)

        #     plt.imshow(img)
        #     plt.show()

        result = pytesseract.image_to_string(img, config='--psm 6')
        result = result.strip()

        def try_convert_numeric(x):
            if re.match(r'[a-zA-Z]', x):
                return x

            numeric = re.sub(r'[^0123456789.]', '', x)
            try:
                if '.' in numeric:
                    numeric = float(numeric)
                else:
                    numeric = int(numeric)
            except ValueError:
                return x
            return numeric

        plot_expected_type_map = {
            'vertical_bar': {
                'x-axis': ['categorical', 'numeric'],
                'y-axis': 'numeric'
            },
            'horizontal_bar': {
                'x-axis': 'numeric',
                'y-axis': 'categorical'
            },
            'dot': {
                'x-axis': ['categorical', 'numeric'],
                'y-axis': ['numeric', None]
            },
            'line': {
                'x-axis': ['categorical', 'numeric'],
                'y-axis': 'numeric'
            },
            'scatter': {
                'x-axis': 'numeric',
                'y-axis': 'numeric'
            }

        }

        expected_type = plot_expected_type_map[plot_type][axis_name]
        if expected_type == 'numeric' or 'numeric' in expected_type:
            result = try_convert_numeric(x=result)

        return result

    @staticmethod
    def _rotate_cropped_text(img: np.ndarray, mask):
        mask = mask.numpy().astype('uint8')
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

        def find_largest_mask(contours):
            """sometimes the mask gets split up. take the largest."""
            largest_idx = None
            largest_area = -float('inf')
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > largest_area:
                    largest_idx = i
                    largest_area = w * h
            return largest_idx

        if len(contours) == 0:
            raise RuntimeError('found no contours')
        elif len(contours) > 1:
            contour = contours[find_largest_mask(contours=contours)]
        else:
            contour = contours[0]

        center, size, angle = cv2.minAreaRect(contour)
        x, y, w, h = cv2.boundingRect(contour)
        #     print(w, h)
        if h > 2 * w and angle == 90:
            # probably vertical text and angle should be 0
            angle = 0

        #     plt.imshow(mask, cmap='gray')
        #     plt.show()

        mask = TF.rotate(Image.fromarray(mask), angle - 90, expand=True)
        mask = np.array(mask)
        #     plt.imshow(mask, cmap='gray')
        #     plt.show()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

        if len(contours) > 1:
            contour = contours[find_largest_mask(contours=contours)]
        else:
            contour = contours[0]

        x, y, w, h = cv2.boundingRect(contour)

        img = Image.fromarray(img.moveaxis(0, 2).numpy())
        img = TF.rotate(img, angle - 90,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        expand=True)
        img = np.array(img)

        rotated_img = img[y:y + h, x:x + w]
        return rotated_img
