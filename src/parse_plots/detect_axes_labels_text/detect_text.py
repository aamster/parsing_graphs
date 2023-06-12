import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np
import pandas as pd
import pytesseract
import torch
from PIL import Image
from easyocr.recognition import get_text
from easyocr.utils import get_image_list, reformat_input, get_paragraph
from sklearn.linear_model import LinearRegression
from torchvision import datapoints
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

from parse_plots.find_axes_tick_labels.dataset import axes_label_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


class DetectText:
    def __init__(
        self,
        axes_segmentations: Dict,
        images_dir: Path,
        plot_types: Dict[str, str],
        easyocr_reader,
        segmentation_resize=(448, 448),
    ):
        self._axes_segmentations = axes_segmentations
        self._images_dir = images_dir
        self._segmentation_resize = segmentation_resize
        self._plot_types = plot_types
        self._easyocr_reader = easyocr_reader

    def run(self):
        res = {}
        if len(self._axes_segmentations) > 1:
            iterable = tqdm(self._axes_segmentations.items(),
                            total=len(self._axes_segmentations))
        else:
            iterable = self._axes_segmentations.items()
        for file_id, pred in iterable:
            img = Image.open(f'{self._images_dir / file_id}.jpg')
            img = datapoints.Image(img)

            boxes = pred['boxes']
            masks = pred['masks']
            labels = pred['labels']

            boxes, masks = self._resize_boxes_and_masks(
                img=img,
                boxes=boxes,
                masks=masks)

            cropped_images = [
                self._preprocess_cropped_text(img=img, mask=mask)
                for mask in masks]

            text = self._get_text(imgs=cropped_images)
            text = np.array(text)

            x_axis_text = text[torch.where(labels == 1)[0]]
            y_axis_text = text[torch.where(labels == 2)[0]]
            axis_text = {
                'x-axis': x_axis_text[sort_boxes(boxes[torch.where(labels == 1)[0]], axis='x-axis')],
                'y-axis': y_axis_text[sort_boxes(boxes[torch.where(labels == 2)[0]], axis='y-axis')],

            }

            res[file_id] = {
                axis: self._postprocess_text(
                    axis=axis,
                    plot_type=self._plot_types[file_id],
                    axis_text=axis_text[axis]
                ) for axis in ('x-axis', 'y-axis')}
        return res

    def _postprocess_text(
        self,
        axis_text,
        axis,
        plot_type
    ):
        expected_type = plot_expected_type_map[plot_type][axis]
        if expected_type == 'numeric' or 'numeric' in expected_type:
            axis_text = [try_convert_numeric(x=x) for x in axis_text]
            numeric_frac = sum([isinstance(x, (int, float)) for x in axis_text]) \
                / len(axis_text)
            # if it is numeric or more than half are numeric, assume it is
            # numeric
            if expected_type == 'numeric' or numeric_frac > 0.5:
                # protect against all string values
                if all(isinstance(x, str) for x in axis_text):
                    pass
                else:
                    # If not numeric, set to null
                    axis_text = [x if isinstance(x, (int, float)) else np.nan for x in axis_text]

                    # TODO there's too many edge cases to get this right.
                    # maybe better to just finetune ocr model
                    try:
                        axis_text = self._correct_numeric_sequence(axis=axis_text)
                    except Exception as e:
                        logger.error(e)
                        axis_text = np.array(axis_text)
                        axis_text[np.isnan(axis_text)] = 0
                        axis_text = axis_text.tolist()
        return axis_text

    def _resize_boxes_and_masks(self, img, boxes, masks):
        boxes = datapoints.BoundingBox(
            boxes,
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=self._segmentation_resize,
            dtype=torch.float
        )
        masks = datapoints.Mask(masks, dtype=torch.uint8)

        boxes, masks = transforms.Resize(
            size=img.shape[1:],
            antialias=True)(boxes, masks)
        return boxes, masks

    @staticmethod
    def _get_oriented_bboxes(
        masks: torch.tensor
    ):
        oriented_bboxes = []
        for mask in masks:
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

            rect = cv2.minAreaRect(contour)
            points = cv2.boxPoints(rect)
            oriented_bboxes.append(points.astype('int'))
        return oriented_bboxes

    def _preprocess_cropped_text(
        self,
        img: torch.tensor,
        mask: torch.tensor
    ) -> np.array:
        img = self._rotate_cropped_text(img=img, mask=mask)

        def invert_background(img):
            """tesseract does poorly when bg is not white.
            inverts img so that the background is white"""
            return np.array(TF.invert(Image.fromarray(img)))

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
        return img

    def _get_text(
        self,
        imgs: List[np.array]
    ):
        #     plt.imshow(img)
        #     plt.show()

        #result = pytesseract.image_to_string(img, config='--psm 6')
        # TODO recognize gives bad results
        # manually preprocess, and give it to get_text

        preprocessed_images = []
        max_widths = []
        for img in imgs:
            _, img = reformat_input(img)
            y_max, x_max = img.shape
            image_list, max_width = get_image_list(
                horizontal_list=[[0, x_max, 0, y_max]],
                free_list=[],
                img=img
            )
            preprocessed_images.append(image_list[0])
            max_widths.append(max_width)

        results = get_text(
            character=self._easyocr_reader.character,
            imgH=64,
            imgW=int(max(max_widths)),
            recognizer=self._easyocr_reader.recognizer,
            converter=self._easyocr_reader.converter,
            image_list=preprocessed_images,
            device=self._easyocr_reader.device,
            batch_size=64,
            workers=0
        )
        results = [item[1] for item in results]
        # results = self._easyocr_reader.recognize(
        #     img,
        #     free_list=oriented_bboxes,
        #     horizontal_list=[],
        #     batch_size=64,
        #     detail=0
        # )
        # for i in range(len(results)):
        #     results[i][1] = results[i][1].strip()
        return results

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

    @staticmethod
    def _correct_numeric_sequence(axis: List[Union[int, float, str]]):
        """tries to correct bad ocr
        1. find most frequent diff
        2. Assume that numbers should be evenly spaced, create sequence with
            that diff
        """
        axis = np.array(axis)
        diff = pd.Series(axis).diff()
        expected_diff = diff.mode().iloc[0]

        start = axis[np.where(diff == expected_diff)[0][0]] - expected_diff * \
                np.where(diff == expected_diff)[0][0]
        end = axis[np.where(diff == expected_diff)[0][0]] + expected_diff * (
                    len(axis) - np.where(diff == expected_diff)[0][0])
        interpolated = np.arange(start, end, expected_diff)
        if (axis[~np.isnan(axis)].astype('int') == axis[~np.isnan(axis)]).all():
            interpolated = interpolated.astype('int')
        return interpolated.tolist()


def try_convert_numeric(x) -> Union[int, float, str]:
    valid_comma = re.search(r',\d{3}', x) is not None
    if ',' in x and not valid_comma:
        # assume it is a dot ?
        x = x.replace(',', '.')
    numeric = re.sub(r'[^-—0123456789.]', '', x)
    numeric = re.sub('—', '-', numeric)

    if '-' in numeric and not numeric.startswith('-'):
        # invalid - sign
        numeric = numeric.replace('-', '')

    if len(numeric) == 0:
        # all non-numeric
        return x

    if re.search(r'\d', x) is None:
        # no numbers
        return x

    if '.' in numeric:
        numeric = float(numeric)
    else:
        numeric = int(numeric)
    return numeric


def sort_boxes(boxes, axis):
    box_sort_vals = torch.tensor(
        [box[0].item() if axis == 'x-axis' else -box[3].item() for box
         in boxes])
    sorted_idx = torch.argsort(box_sort_vals)
    return sorted_idx
