import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from easyocr.recognition import get_text
from easyocr.utils import get_image_list, reformat_input
from sklearn.linear_model import LinearRegression

torchvision.disable_beta_transforms_warning()

from torchvision import datapoints, io
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plot_expected_type_map = {
    'vertical_bar': {
        'x-axis': ['categorical'],
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
        'x-axis': ['categorical'],
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
        segmentation_resize=(448, 448)
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

    def run_inference(self):
        axes_segmentations = self._axes_segmentations
        res = {}
        for file_id, pred in tqdm(axes_segmentations.items()):
            axis_text = {}

            img = io.read_image(f'{self._images_dir / file_id}.jpg')
            img = datapoints.Image(img)

            for axis, axis_pred in pred.items():
                boxes = axis_pred['boxes']
                masks = axis_pred['masks']

                if boxes.shape[0] > 0 and masks.shape[0] > 0:
                    boxes, masks = self._resize_boxes_and_masks(
                        img=img,
                        boxes=boxes,
                        masks=masks)

                cropped_images = [
                    self._preprocess_cropped_text(img=img, mask=mask)
                    for mask in masks]

                #######
                # DEBUG
                axis_text[axis] = []
                continue
                ########

                if len(cropped_images) > 0:
                    text = self._get_text(imgs=cropped_images)
                    # remove duplicates
                    text_dups_removes = []
                    text_idxs_kept = []
                    for i, label in enumerate(text):
                        if label not in text_dups_removes:
                            text_dups_removes.append(label)
                            text_idxs_kept.append(i)
                    text = text_dups_removes
                    axis_pred['boxes'] = axis_pred['boxes'][text_idxs_kept]
                    axis_pred['masks'] = axis_pred['masks'][text_idxs_kept]
                    axis_pred['labels'] = axis_pred['labels'][text_idxs_kept]

                else:
                    text = []
                text = np.array(text)
                axis_text[axis] = text
            res[file_id] = axis_text

            res[file_id] = {
                axis: self._postprocess_text(
                    axis=axis,
                    plot_type=self._plot_types[file_id],
                    axis_text=axis_text[axis]
                ) if len(axis_text[axis]) > 0 else []
                for axis in ('x-axis', 'y-axis')}
        return res

    def _postprocess_text(
        self,
        axis_text,
        axis,
        plot_type
    ):
        def is_numeric(x):
            try:
                float(x)
                is_float = True
            except ValueError:
                is_float = False
            return is_float

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
                elif all(isinstance(x, (int, float)) for x in axis_text):
                    pass
                else:
                    # If not numeric, interpolate
                    reg = LinearRegression()
                    numeric_text_idx = [
                        i for i in range(len(axis_text))
                        if isinstance(axis_text[i], (int, float))]
                    non_numeric_text = [i for i in range(len(axis_text))
                                        if i not in numeric_text_idx]
                    reg.fit(
                        np.array(numeric_text_idx).reshape(-1, 1),
                        np.array(axis_text)[numeric_text_idx].reshape(-1, 1))
                    preds = reg.predict(
                        np.array(non_numeric_text).reshape(-1, 1)).flatten()
                    preds = dict(zip(non_numeric_text, preds))
                    axis_text = [
                        axis_text[i] if i in numeric_text_idx else preds[i]
                        for i in range(len(axis_text))]

        if all(is_numeric(x) for x in axis_text):
            axis_text = self._correct_numeric_sequence(axis=axis_text)

        if expected_type == 'categorical' or 'categorical' in expected_type:
            # make sure all string and no numbers
            mode = Counter([type(x) for x in axis_text]).most_common(n=1)[0][0]
            axis_text = [x if isinstance(x, mode) else mode(x)
                         for x in axis_text]
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
        return img

    def _get_text(
        self,
        imgs: List[np.array]
    ):
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
            imgH=32,
            imgW=100,
            recognizer=self._easyocr_reader.recognizer,
            converter=self._easyocr_reader.converter,
            image_list=preprocessed_images,
            device=self._easyocr_reader.device,
            batch_size=64,
            workers=0
        )
        results = [item[1] for item in results]
        return results

    @staticmethod
    def _rotate_cropped_text(img: torch.Tensor, mask):
        rect = DetectText.get_min_area_rect(mask=mask)
        if rect is None:
            return img
        center, size, angle = rect

        mask = TF.rotate(mask.unsqueeze(dim=0), angle - 90, expand=True)
        mask = mask.moveaxis(0, 2).numpy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

        if len(contours) > 1:
            contour = contours[DetectText.find_largest_mask(contours=contours)]
        elif len(contours) == 1:
            contour = contours[0]
        else:
            return img

        #########
        # DEBUG
        return img
        #########

        x, y, w, h = cv2.boundingRect(contour)

        img = TF.rotate(Image.fromarray(img.moveaxis(0, 2).numpy()), angle - 90,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        expand=True)

        img = np.array(img)
        rotated_img = img[y:y + h, x:x + w]
        return rotated_img

    @staticmethod
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

    @staticmethod
    def get_min_area_rect(mask: torch.Tensor):
        mask = mask.numpy().astype('uint8')
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            return None
        elif len(contours) > 1:
            contour = contours[DetectText.find_largest_mask(contours=contours)]
        else:
            contour = contours[0]

        center, size, angle = cv2.minAreaRect(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if h > 1.5 * w:
            if angle == 90:
                # probably vertical text and angle should be 0
                angle = 0
        else:
            if angle == 0:
                # probably horizontal text. angle should be 90
                angle = 90

        return center, size, angle

    @staticmethod
    def _correct_numeric_sequence(axis: List[Union[int, float, str]]):
        """tries to correct bad ocr
        1. find most frequent diff
        2. Assume that numbers should be evenly spaced, create sequence with
            that diff
        """
        axis_float = np.array([float(x) for x in axis])
        diff = pd.Series(axis_float).diff()
        expected_diff = diff.mode().iloc[0]

        start = axis_float[np.where(diff == expected_diff)[0][0]] - expected_diff * \
                np.where(diff == expected_diff)[0][0]
        end = axis_float[np.where(diff == expected_diff)[0][0]] + expected_diff * (
                    len(axis_float) - np.where(diff == expected_diff)[0][0])
        interpolated = np.arange(start, end, expected_diff)
        if (axis_float[~np.isnan(axis_float)].astype('int') == axis_float[~np.isnan(axis_float)]).all():
            interpolated = interpolated.astype('int')

        if str(axis[0]) == axis[0]:
            interpolated = np.array([str(x) for x in interpolated])
        return interpolated.tolist()


def try_convert_numeric(x) -> Union[int, float, str]:
    # if there are any letters, don't convert
    if re.findall(r'[a-zA-Z]', x):
        return x

    if '-' in x and x.index('-') > 0:
        # "-" in the string and it's in the middle
        # assume it's a string
        return x

    dots = [v for v in x if v == '.']
    if len(dots) > 1:
        # european comma?
        # like 7.000.000
        x = x.replace('.', '')

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
