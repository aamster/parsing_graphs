import json
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Literal, Optional

import cv2
import numpy as np
import torch.utils.data
from sklearn.cluster import KMeans
from torch import Tensor
from torch.utils.data.dataset import T_co
from torchvision import datapoints, io
from torchvision.utils import draw_keypoints
from torchvision.transforms.v2 import functional as F


from parse_plots.utils import string_to_float, resize_plot_bounding_box


class BadDataError(RuntimeError):
    pass


plot_type_id_map = {
    'dot': 1,
    'horizontal_bar': 2,
    'vertical_bar': 3,
    'line': 4,
    'scatter': 5
}

plot_type_id_value_map = {
    1: 'dot',
    2: 'horizontal_bar',
    3: 'vertical_bar',
    4: 'line',
    5: 'scatter'
}


class DetectPlotValuesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        plot_ids: List[str],
        plots_dir,
        transform,
        annotations_dir: Optional[Path] = None,
        is_train: bool = True,
        plot_meta: Optional[Dict[str, Dict]] = None
    ):
        super().__init__()
        bad_ids = [
            '835d25a0429f',
            '416112b0a9df',
            'aaed9df80c3e',
            'e012ac3c181c',
            'aa9df520a5f2',
            '6ce4bc728dd5',
            '04296b42ba61',
            '733b9b19e09a',
            'd7c6e100bfd7',
            'd75a8f2a99db',
            'eac402fad0a0',
            'e0eb4bd323a1',
            'f3e97702514d',
            '87963c7d66e3',
            'b6f5fbdf7bc9',
            '7b79ba7b9ae8',
            '4a3095dcc9fc',
            '2343fb5ef761',
            'a9ffeb1e41e2',
            '11027193a81a',
            '2e8a9140b20c',
            '43f6a39f9750',
            'ac44fbe3135e',
            '3968efe9cbfc',
            'ece590cebb43',
            '2deb0dd554d4',
            'd0cf883b1e13',
            '3b99e5248fb1',
            'bea99357d19c',
            '6a92d147f4d5',
            'dc03a1353258',
            '6b543e5debb7',
            'a9e7b2b46569',
            '496de625e57a',
            '18204b9effce',
            '288b6166f5bc',
            '2ff5d2491913',
            '7c1346d05b63',
            'fa91f20f277d',
            'e28dc785e713',
            '12df74ce02de',
            'a613be731d61',
            '6771e4a4fab5',
            '6447c2a5e487',
            '60923b97d2b5',
            '9affc9b7cb76',
            'c25da96d5aaf',
            '8e71f5f4f2d2',
            '4163f70a77b3',
            '62a5fe77db68',
            '353fab6e4d7a',
            'd2a1ae5b168f',
            '8c5e7b72d028',
            '4d4dd29a2ee6',
            '89d24be7fcb0',
            'abc2c9825279',
            '31e4db3c76d5',
            'e2ee063cb374',
            '7bb3684ac793',
            '8b17ddaec807',
            '388757b10ad4',
            '58b3a206d02a',
            '2e00ba9ef727',
            '7bdbafd2cb3e',
            'ed5d7906928c',
            'd7f597a3b9c9'
        ]
        if is_train:
            plot_ids = [x for x in plot_ids if x not in bad_ids]
            if annotations_dir is None:
                raise ValueError('annotations_dir must be given if train')
        self._plot_files = [f'{x}.jpg' for x in plot_ids]
        self._plots_dir = Path(plots_dir)
        self._annotations_dir = Path(annotations_dir) \
            if annotations_dir is not None else None
        self._transform = transform
        self._is_train = is_train
        self._plot_meta = plot_meta

    def __getitem__(self, index) -> T_co:
        id = Path(self._plot_files[index]).stem
        img = io.read_image(str(self._plots_dir / f'{id}.jpg'))
        img = datapoints.Image(img)

        if not self._is_train:
            if self._plot_meta is None:
                raise ValueError('plot_meta must be given if not train')
            plot_bb = self._plot_meta[id]['plot_bbox']
            plot_bb = resize_plot_bounding_box(
                img=img,
                plot_bounding_box=plot_bb
            )
            if self._transform is not None:
                img = self._transform(img)

            return img, {
                'image_id': id,
                'plot_bbox': plot_bb
            }

        with open(self._annotations_dir / f'{id}.json') as f:
            a = json.load(f)

        if a['chart-type'] == 'horizontal_bar':
            # horizontal bar has axes mixed up
            axes = deepcopy(a['axes'])
            axes['y-axis'] = a['axes']['x-axis']
            axes['x-axis'] = a['axes']['y-axis']
            a['axes'] = axes

        if a['chart-type'] == 'dot':
            img = img[:, a['plot-bb']['y0']-10:a['plot-bb']['y0'] + a['plot-bb'][
                'height']+10,
                  a['plot-bb']['x0']:a['plot-bb']['x0'] + a['plot-bb'][
                      'width']]

        if a['chart-type'] == 'line':
            mask = get_targets(img=img, a=a)
            mask = datapoints.Mask(mask)
            if self._transform is not None:
                img, mask = self._transform(
                    img, mask)

            target = {
                'image_id': id,
                'mask': mask,
                'plot_bbox': a['plot-bb']
            }
            return img, target
        else:
            bboxes = get_targets(img=img, a=a)

            labels = torch.tensor([
                plot_type_id_map[a['chart-type']] for _ in range(len(bboxes))])

            if self._transform is not None:
                img, bboxes, labels = self._transform(
                    img, bboxes, labels)

            target = {
                'image_id': id,
                'chart_type': a['chart-type'],
                'boxes': bboxes,
                'labels': labels
            }

            return img, target

    def __len__(self):
        return len(self._plot_files)

    @staticmethod
    def convert_keypoints_to_mask(
            keypoints,
            img_shape,
            keypoint_as_circle: bool = False,
            keypoint_circle_radius: int = 10
    ):
        if keypoint_as_circle:
            masks_rgb = torch.zeros(len(keypoints), 3, *img_shape, dtype=torch.uint8)
            masks = torch.zeros(len(keypoints), *img_shape, dtype=torch.uint8)
        else:
            masks = torch.zeros(len(keypoints), *img_shape, dtype=torch.uint8)
            masks_rgb = None
        keypoints = keypoints.int()

        for i, keypoint in enumerate(keypoints):
            if keypoint_as_circle:
                keypoint = keypoint.unsqueeze(dim=0)
                masks[i] = draw_keypoints(
                    image=masks_rgb[i],
                    keypoints=keypoint,
                    radius=keypoint_circle_radius,
                    colors=(1, 1, 1)
                )[0]

            else:
                masks[i][keypoint[1], keypoint[0]] = 1
        return masks


def get_targets(img, a: Dict):
    if a['chart-type'] == 'line':
        cropped_img = img[:, a['plot-bb']['y0']+10:a['plot-bb']['y0'] + a['plot-bb'][
            'height']-10,
              a['plot-bb']['x0']+10:a['plot-bb']['x0'] + a['plot-bb'][
                  'width']-10]
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(
            cropped_img.moveaxis(0, 2).reshape(-1, 3))
        cropped_mask = kmeans.labels_.reshape(cropped_img.shape[1:])
        cropped_mask = torch.tensor(cropped_mask, dtype=torch.uint8)
        zero_count = (kmeans.labels_ == 0).sum()
        one_count = (kmeans.labels_ == 1).sum()
        line_label = 1 if one_count < zero_count else 0
        if line_label == 0:
            # reverse the labels so mask selects the line
            cropped_mask_ = torch.zeros_like(cropped_mask)
            cropped_mask_[cropped_mask == 0] = 1
            cropped_mask_[cropped_mask == 1] = 0
            cropped_mask = cropped_mask_

        mask = torch.zeros(*img.shape[1:], dtype=torch.uint8)
        mask[a['plot-bb']['y0']+10:a['plot-bb']['y0'] + a['plot-bb'][
            'height']-10, a['plot-bb']['x0']+10:a['plot-bb']['x0'] + a['plot-bb'][
                  'width']-10] = cropped_mask

        return mask
    elif a['chart-type'] == 'dot':
        x_pts, y_pts, dot_width, dot_height = get_points(img=img, a=a)
        bboxes = get_bounding_boxes(
            img=img,
            a=a,
            x_pts=x_pts,
            y_pts=y_pts,
            dot_width=dot_width,
            dot_height=dot_height
        )
    else:
        x_pts, y_pts = get_points(img=img, a=a)
        bboxes = get_bounding_boxes(img=img, a=a, x_pts=x_pts, y_pts=y_pts)
    return bboxes


def get_bboxes_for_dot(a: Dict, x_pts, y_pts, dot_width, dot_height):
    pts = list(zip(x_pts, y_pts))
    boxes = []
    plot_bb = a['plot-bb']
    for tick in a['axes']['x-axis']['ticks']:
        tick = tick['tick_pt']

        # move tick over since we are only looking at the area within plot_bb
        tick['x'] -= plot_bb['x0']

        dots = []
        for pt in pts:
            if tick['x'] - dot_width / 2 <= pt[0] <= tick['x'] + dot_width / 2:
                dots.append(pt)
        dots = np.array(dots)
        if dots.shape[0] == 0:
            # not dots at this tick
            continue
        box_height = max(0, dots[:, 1].min() - dot_height)
        box_width = dot_width
        x_min = tick['x'] - dot_width / 2
        y_min = box_height

        boxes.append([x_min, y_min, x_min + box_width, plot_bb['height']])
    return boxes


def get_bounding_boxes(img, a, x_pts, y_pts, **kwargs):
    pts = zip([int(x) for x in x_pts], [int(y) for y in y_pts])
    bboxes = []

    for i, (x, y) in enumerate(pts):
        if a['chart-type'] == 'vertical_bar':
            max_y_tick_pos = max(
                [x['tick_pt']['y'] for x in a['axes']['y-axis']['ticks']])
            x_diff = a['axes']['x-axis']['ticks'][1]['tick_pt']['x'] - \
                     a['axes']['x-axis']['ticks'][0]['tick_pt']['x']
            box = [x - int(x_diff/2), y, x + int(x_diff/2), max_y_tick_pos]
        elif a['chart-type'] == 'horizontal_bar':
            min_x_tick_pos = min(
                [x['tick_pt']['x'] for x in a['axes']['x-axis']['ticks']])
            y_diff = a['axes']['y-axis']['ticks'][1]['tick_pt']['y'] - \
                     a['axes']['y-axis']['ticks'][0]['tick_pt']['y']
            box = [min_x_tick_pos, y - int(y_diff/2), x, y + int(y_diff/2)]
        elif a['chart-type'] == 'dot':
            box = [
                x-int(kwargs['dot_width']/2),
                y-int(kwargs['dot_height']/2),
                x+int(kwargs['dot_width']/2),
                y+int(kwargs['dot_height']/2)
            ]
        elif a['chart-type'] == 'scatter':
            box = [x-8, y-8, x+8, y+8]
        else:
            raise NotImplementedError
        bboxes.append(box)
    bboxes = datapoints.BoundingBox(
        np.array(bboxes),
        format=datapoints.BoundingBoxFormat.XYXY,
        spatial_size=F.get_spatial_size(img),
        dtype=torch.float
    )
    return bboxes


def get_points(a, img: Optional[Tensor] = None):
    if a['chart-type'] == 'line':
        x_pts, y_pts = get_line_points(a=a)
    elif a['chart-type'] == 'scatter':
        x_pts, y_pts = get_scatter_points(a=a)
    elif a['chart-type'] == 'dot':
        if img is None:
            raise ValueError('must provide img if dot')
        x_pts, y_pts, dot_width, dot_height = get_dot_points(img=img)
        return x_pts, y_pts, dot_width, dot_height
    else:
        x_pts, y_pts = get_bar_points(a=a, bar_type=a['chart-type'])
    return x_pts, y_pts


def interpolate_points(a: Dict, axis: str):
    pts = []

    for i, val in enumerate(a['data-series']):
        axis_val = val[axis]

        # adding small number so no division by 0
        axis_val += 1e-9

        axis_ids = set([x['id'] for x in a['axes'][f'{axis}-axis']['ticks']])
        if len(axis_ids) == 0:
            raise BadDataError('There are no tick marks in the data!!')

        tick_labels = [{
            'id': x['id'],
            'val': string_to_float(x['text'])
        } for x in a['text'] if x['id'] in axis_ids]

        closest_label = None
        least_diff = float('inf')
        for tick_label in tick_labels:
            diff = abs(axis_val - tick_label['val'])
            if diff < least_diff:
                least_diff = diff
                closest_label = tick_label

        closest_axis_mark = [x for x in a['axes'][f'{axis}-axis']['ticks']
                             if x['id'] == closest_label['id']][0]

        axis_spacing = a['axes'][f'{axis}-axis']['ticks'][1]['tick_pt'][axis] - \
                         a['axes'][f'{axis}-axis']['ticks'][0]['tick_pt'][axis]
        axis_diff = abs(tick_labels[0]['val'] - tick_labels[1]['val'])
        if axis_diff == 0:
            raise BadDataError('The axis value repeats!!')

        pt = closest_axis_mark['tick_pt'][axis]
        translation = (
                axis_spacing *
                abs(axis_val - closest_label['val']) /
                axis_diff
        )
        if axis == 'y':
            if axis_val < closest_label['val']:
                pt += translation
            else:
                pt -= translation
        else:
            if axis_val < closest_label['val']:
                pt -= translation
            else:
                pt += translation
        pts.append(pt)
    return pts


def get_line_points(a: Dict):
    y_pts = interpolate_points(a=a, axis='y')

    id_text_map = {x['id']: x['text'] for x in a['text']}
    x_data_series = set([x['x'] for x in a['data-series']])
    x_pts = [x for x in a['axes']['x-axis']['ticks']]

    # limit x_pts to only those found in data-series
    x_pts = [x for x in x_pts if id_text_map[x['id']] in x_data_series]

    x_pts = [x['tick_pt']['x'] for x in x_pts]

    return x_pts, y_pts


def get_scatter_points(a: Dict):
    y_pts = interpolate_points(a=a, axis='y')
    x_pts = interpolate_points(a=a, axis='x')

    return x_pts, y_pts


def get_bar_points(a: Dict, bar_type: Literal['vertical_bar', 'horizontal_bar']):
    if bar_type == 'vertical_bar':
        y_points = interpolate_points(a=a, axis='y')
        x_points = [x['tick_pt']['x'] for x in a['axes']['x-axis']['ticks']]
    else:
        x_points = interpolate_points(a=a, axis='x')
        y_points = [x['tick_pt']['y'] for x in a['axes']['y-axis']['ticks']]
    return x_points, y_points


def get_dot_points(img: torch.tensor):
    contours = get_plot_element_contours(img=img)
    x_pts = []
    y_pts = []
    dot_width = None
    dot_height = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_pts.append(x + int(w / 2))
        y_pts.append(y + int(h / 2))
        dot_width = w
        dot_height = h
    return x_pts, y_pts, dot_width, dot_height


def get_plot_element_contours(img: torch.tensor):
    img = img.moveaxis(0, 2).numpy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, bin_img = cv2.threshold(gray,
                                 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img,
                               cv2.MORPH_OPEN,
                               kernel,
                               iterations=2)
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)

    ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255,
                                 cv2.THRESH_BINARY)

    sure_fg = sure_fg.astype(np.uint8)
    sure_bg = cv2.dilate(bin_img, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)

    # background becomes 1
    markers += 1

    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    labels = np.unique(markers)
    all_contours = []
    for label in labels[2:]:
        # Create a binary image in which only the area of the label is in
        # the foreground
        # and the rest of the image is in the background
        target = np.where(markers == label, 255, 0).astype(np.uint8)

        # Perform contour extraction on the created binary image
        contours, hierarchy = cv2.findContours(
            target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        all_contours.append(contours[0])
    return all_contours
