import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List, Any

import albumentations
import argschema as argschema
import cv2
import easyocr
import numpy as np
import pandas as pd
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from backboned_unet import Unet
from lightning import Trainer, LightningModule
from torchvision.models.segmentation import FCN_ResNet50_Weights

from parse_plots.data_module import PlotDataModule
from parse_plots.detect_plot_values.dataset import DetectPlotValuesDataset
from parse_plots.detect_plot_values.line_segmentation_model import \
    SegmentLinePlotModel
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b1

torchvision.disable_beta_transforms_warning()

import torchvision.transforms.v2 as T
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, \
    fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from parse_plots.classify_plot_type.dataset import ClassifyPlotTypeDataset
from parse_plots.classify_plot_type.model import ClassifyPlotTypeModel
from parse_plots.detect_axes_labels_text.detect_text import DetectText, \
    sort_boxes
from parse_plots.detect_plot_values.model import DetectPlotValuesModel
from parse_plots.find_axes_tick_labels.dataset import FindAxesTickLabelsDataset
from parse_plots.find_axes_tick_labels.model import SegmentAxesTickLabelsModel


class ParsePlotsSchema(argschema.ArgSchema):
    plots_dir = argschema.fields.InputDir(required=True)
    classify_plot_type_checkpoint = argschema.fields.InputFile(required=True)
    segment_axes_tick_labels_checkpoint = argschema.fields.InputFile(required=True)
    ocr_model_storage_directory = argschema.fields.InputDir(required=True)
    ocr_user_network_directory = argschema.fields.InputDir(required=True)
    line_plot_segmentation_checkpoint = argschema.fields.InputFile(required=True)
    plot_object_detector_checkpoint = argschema.fields.InputFile(required=True)
    out_dir = argschema.fields.OutputDir(required=True)
    is_debug = argschema.fields.Boolean(default=False)
    batch_size = argschema.fields.Int(default=8)
    debug_num = argschema.fields.Int(default=256, required=False)


class ParsePlotsRunner(argschema.ArgSchemaParser):
    default_schema = ParsePlotsSchema

    def __init__(self, input_data=None, args=None):
        super().__init__(
            schema_type=ParsePlotsSchema,
            input_data=input_data,
            args=args
        )
        plot_files = os.listdir(self.args['plots_dir'])
        plot_ids = [Path(x).stem for x in plot_files]
        if self.args['is_debug']:
            plot_ids = plot_ids[:self.args['debug_num']]
            #plot_ids = ['000b92c3b098']
        self._plot_ids = plot_ids
        self._is_debug = self.args['is_debug']
        self._segment_line_plot_model = \
            SegmentLinePlotModel.load_from_checkpoint(
                checkpoint_path=self.args['line_plot_segmentation_checkpoint'],
                learning_rate=1e-3,
                model=Unet(classes=1, pretrained=False),
                hyperparams={},
                map_location=(
                    torch.device('cpu') if not torch.has_cuda else None)
            )
        self._detect_plot_values_model = \
            DetectPlotValuesModel.load_from_checkpoint(
                checkpoint_path=self.args['plot_object_detector_checkpoint'],
                learning_rate=1e-3,
                model=self.detect_plot_values_model,
                hyperparams={},
                map_location=(
                    torch.device('cpu') if not torch.has_cuda else None)
            )
        self._classify_plot_type_model = self.classify_plot_type_model
        self._detect_axes_labels_model = self.detect_axes_labels_model
        self._trainer = Trainer()

    @property
    def detect_plot_values_model(self):
        model = fasterrcnn_resnet50_fpn_v2(weights=None)

        # replace box classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 6)
        return model

    @property
    def classify_plot_type_model(self):
        architecture = efficientnet_b1()

        architecture.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1280, 5)
        )

        model: ClassifyPlotTypeModel = \
            ClassifyPlotTypeModel.load_from_checkpoint(
                checkpoint_path=self.args['classify_plot_type_checkpoint'],
                learning_rate=1e-3,
                model=architecture,
                hyperparams={},
                map_location=(torch.device('cpu') if not torch.has_cuda
                              else None)
            )
        return model

    @property
    def detect_axes_labels_model(self):
        model = maskrcnn_resnet50_fpn_v2(weights=None)

        # replace box classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)

        # replace mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask\
            .in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           3)
        model: SegmentAxesTickLabelsModel = \
            SegmentAxesTickLabelsModel.load_from_checkpoint(
                checkpoint_path=(
                    self.args['segment_axes_tick_labels_checkpoint']),
                learning_rate=1e-3,
                model=model,
                hyperparams={},
                map_location=(torch.device('cpu') if not torch.has_cuda
                              else None)
        )
        return model

    def run(self):
        all_data_series = []
        all_axes_segmentations = self._find_axes_tick_labels()
        n_batches = math.ceil(len(self._plot_ids) / self.args['batch_size'])
        for batch_idx, axes_segmentations in enumerate(all_axes_segmentations,
                                                       start=1):
            start = time.time()
            self.logger.info(
                f'{batch_idx}/{n_batches} Getting plot values for '
                f'{list(axes_segmentations.keys())}')
            plot_types = self._classify_plot_type(
                axes_segmentations=axes_segmentations
            )

            # Fix horizontal bar axes segmentations
            horizontal_bar_plots = {
                k: v for k, v in plot_types.items() if v == 'horizontal_bar'}
            if horizontal_bar_plots:
                hb_axes_segmentations = \
                    self._find_axes_tick_labels_for_horizontal_bar(
                        plot_ids=[k for k in horizontal_bar_plots]
                    )
                for img_id in hb_axes_segmentations:
                    if hb_axes_segmentations[img_id] is not None:
                        axes_segmentations[img_id]['y-axis']['boxes'] = \
                            hb_axes_segmentations[img_id]['bboxes']
                        axes_segmentations[img_id]['y-axis']['masks'] = \
                            hb_axes_segmentations[img_id]['masks']

            tick_labels = self. _detect_axes_label_text(
                axes_segmentations=axes_segmentations,
                plot_types=plot_types
            )

            plot_values = self._detect_plot_values(
                axes_segmentations=axes_segmentations,
                plot_types=plot_types,
                tick_labels=tick_labels
            )

            # ##########
            # # DEBUG
            # ##########
            # data_series = self._construct_data_series(
            #     plot_types={k: 'vertical_bar' for k in axes_segmentations},
            #     file_id_plot_values_map={k: [('abc', 0.0), ('def', 1.0)]
            #                              for k in axes_segmentations}
            # )
            # data_series = pd.DataFrame(data_series)
            # all_data_series.append(data_series)
            # continue
            # ##########
            # # END DEBUG
            # ##########

            file_id_plot_values_map = {}
            for file_id, plot_points in plot_values.items():
                # add string values
                plot_points_ = []
                for i in range(len(plot_points)):
                    plot_point = [plot_points[i][0], plot_points[i][1]]
                    if plot_point[0] is None:
                        if i >= len(tick_labels[file_id]['x-axis']):
                            # more plot values than ticks (in the case of bar
                            # chart is a problem)
                            plot_point[0] = ''
                        else:
                            plot_point[0] = tick_labels[file_id]['x-axis'][i]

                    if plot_point[1] is None:
                        if i >= len(tick_labels[file_id]['y-axis']):
                            # more plot values than ticks (in the case of bar
                            # chart is a problem)
                            plot_point[1] = ''
                        else:
                            plot_point[1] = tick_labels[file_id]['y-axis'][i]

                    plot_points_.append(tuple(plot_point))

                if plot_types[file_id] == 'horizontal_bar':
                    plot_points_ = list(zip(
                        [x[0] for x in plot_points_],
                        [x[1] for x in plot_points_][::-1]))
                file_id_plot_values_map[file_id] = plot_points_

                duration = time.time() - start
                self.logger.info(
                    f'Finished in {duration:.0f} seconds, '
                    f'{(n_batches - batch_idx) * duration:.0f} '
                    f'seconds remaining')
            data_series = self._construct_data_series(
                plot_types=plot_types,
                file_id_plot_values_map=file_id_plot_values_map
            )
            data_series = pd.DataFrame(data_series)
            all_data_series.append(data_series)

        out_path = Path(self.args['out_dir']) / 'submission.csv'
        pd.concat(all_data_series).to_csv(out_path, index=False)
        self.logger.info(f'Wrote submission to {out_path}')

    def _detect_plot_values(
        self,
        axes_segmentations: Dict[str, Dict],
        plot_types: Dict[str, str],
        tick_labels
    ):
        plot_values_img_coordinates = []
        line_plot_ids = [x for x in axes_segmentations
                         if plot_types[x] == 'line']
        if line_plot_ids:
            line_masks = self._detect_plot_values_with_model(
                plot_ids=line_plot_ids,
                plot_types=plot_types,
                model=self._segment_line_plot_model
            )

            for i, plot_id in enumerate(line_plot_ids):
                plot_values_img_coordinates.append((
                    plot_id,
                    self._get_line_plot_values_in_img_coordinates(
                        axes_segmentations=axes_segmentations[plot_id],
                        line_plot_mask=line_masks[0][i]
                    )
                ))

        other_plot_ids = [x for x in axes_segmentations
                          if plot_types[x] != 'line']
        if other_plot_ids:
            predictions = self._detect_plot_values_with_model(
                plot_ids=other_plot_ids,
                plot_types=plot_types,
                model=self._detect_plot_values_model
            )

            predictions = predictions[0]
            for i, plot_id in enumerate(other_plot_ids):
                if plot_types[plot_id] == 'horizontal_bar':
                    if predictions[i]['boxes'].shape[0] > 0:
                        predictions[i]['boxes'] = T.RandomRotation(
                            degrees=(-90, -90))(predictions[i]['boxes'])
                plot_values_img_coordinates.append((
                    plot_id,
                    self._get_plot_values_in_img_coordinates(
                        boxes=predictions[i]['boxes'],
                        plot_type=plot_types[plot_id]
                    )
                ))

        file_id_plot_points_map = {}
        for file_id, img_coordinates in plot_values_img_coordinates:
            axes = self._get_tick_points(
                axes_segmentations=axes_segmentations[file_id]
            )

            if plot_types[file_id] in ('vertical_bar', 'horizontal_bar'):
                # Fix issue with multiple boxes per bar
                img_coordinates = self._remove_duplicate_values_per_bar(
                    plot_values=img_coordinates,
                    plot_type=plot_types[file_id],
                    axes=axes
                )
            plot_points = []
            for coord in img_coordinates:
                plot_point_values = []
                for axis, axis_idx in [('x', 0), ('y', 1)]:

                    if len(axes_segmentations[file_id][f'{axis}-axis']['boxes'])\
                            == 0:
                        plot_point_values.append(None)
                        continue

                    closest_tick_label_idx = \
                        self._get_closest_tick_label_in_image_coordinates(
                            coord=coord,
                            axis=axis,
                            axes_segmentations=axes_segmentations[file_id][f'{axis}-axis']
                        )
                    closest_tick_pt = axes[f'{axis}-axis'][closest_tick_label_idx]
                    closest_tick_val = (
                        tick_labels[file_id][f'{axis}-axis']
                        [closest_tick_label_idx])

                    if isinstance(closest_tick_val, str):
                        plot_point_values.append(None)
                    else:
                        axis_spacing = abs(
                            axes[f'{axis}-axis'][1]['tick_pt'] -
                            axes[f'{axis}-axis'][0]['tick_pt'])
                        axis_diff = abs(
                            tick_labels[file_id][f'{axis}-axis'][0] -
                            tick_labels[file_id][f'{axis}-axis'][1])
                        diff_from_closest_tick_val = \
                            abs(closest_tick_pt['tick_pt'] -
                                coord[axis_idx]) / axis_spacing * axis_diff

                        if axis == 'y':
                            # coordinates increase but values decrease
                            if (coord[axis_idx] >
                                    closest_tick_pt['tick_pt']):
                                plot_val = (closest_tick_val -
                                            diff_from_closest_tick_val)
                            else:
                                plot_val = (closest_tick_val +
                                            diff_from_closest_tick_val)
                        else:
                            # coordinates increase and values increase
                            if (coord[axis_idx] >
                                    closest_tick_pt['tick_pt']):
                                plot_val = (closest_tick_val +
                                            diff_from_closest_tick_val)
                            else:
                                plot_val = (closest_tick_val -
                                            diff_from_closest_tick_val)
                        plot_point_values.append(plot_val)
                plot_points.append(plot_point_values)

            if plot_types[file_id] == 'dot':
                x_axis_numeric = \
                    isinstance(tick_labels[file_id]['x-axis'][0],
                               (int, float))
                dot_counts = defaultdict(int)

                for coord_idx in range(len(plot_points)):
                    x, y = plot_points[coord_idx]

                    if isinstance(x, (int, float)):
                        x = round(x)
                    dot_counts[x] += 1
                if x_axis_numeric:
                    plot_points = [
                        [k, dot_counts[k]] for k in sorted(dot_counts)]
                else:
                    plot_points = [
                        [k, dot_counts[k]] for k in
                        tick_labels[file_id]['x-axis']
                    ]

            if plot_types[file_id] == 'vertical_bar' and \
                    len(tick_labels[file_id]['x-axis']) == \
                    len(plot_points) + 1:
                # it's a histogram
                plot_points = list(zip(
                    tick_labels[file_id]['x-axis'],
                    [x[1] for x in plot_points])) + \
                    [(tick_labels[file_id]['x-axis'][-1],
                      'HISTOGRAM_PLACEHOLDER')]
            file_id_plot_points_map[file_id] = plot_points
        return file_id_plot_points_map

    @staticmethod
    def _get_tick_values(axis: List[Dict], plot_values, axis_name: str):
        """For each tick mark, get the values associated with it
        (found by finding values closest to it)"""
        min_dists = np.zeros(len(plot_values))
        tick_min_dist_idx = np.zeros(len(plot_values))
        tick_values = {x['tick_id']: [] for x in axis}
        for i, (x, y) in enumerate(plot_values):
            plot_value = x if axis_name == 'x-axis' else y
            dists = [abs(plot_value - tick_pt['tick_pt'])
                     for tick_pt in axis]
            min_dist = np.argmin(dists)
            tick_min_dist_idx[i] = min_dist
            min_dists[i] = np.min(dists)

        # protect against a plot value getting assigned to the wrong tick
        # happens in case of bar where we miss the tick mark
        q1, q3 = np.quantile(min_dists, (0.25, 0.75))
        iqr = q3 - q1

        for i, min_dist in enumerate(min_dists):
            if q1 - iqr * 3 <= min_dist <= q3 + iqr * 3:
                tick_values[axis[int(tick_min_dist_idx[i])]['tick_id']]\
                    .append(plot_values[i])

        return tick_values

    def _remove_duplicate_values_per_bar(
            self,
            plot_type: str,
            axes: Dict,
            plot_values: List[Tuple]
    ) -> List[Tuple]:
        """Removes duplicate values per tick for bar. Uses the naive assumption
        that the largest/smallest coord corresponds to the correct box?"""
        values_to_keep = set()
        tick_values = self._get_tick_values(
            axis=(axes['x-axis'] if plot_type == 'vertical_bar'
                  else axes['y-axis']),
            plot_values=plot_values,
            axis_name='x-axis' if plot_type == 'vertical_bar' else 'y-axis'
        )
        for tick_id, values in tick_values.items():
            if not values:
                continue
            if plot_type == 'horizontal_bar':
                values_to_keep.add(
                    sorted(values, key=lambda x: x[0])[-1])
            else:
                values_to_keep.add(
                    sorted(values, key=lambda x: x[1])[0])
        return [x for x in plot_values if x in values_to_keep]

    @staticmethod
    def _get_closest_tick_label_in_image_coordinates(
            coord: Tuple[int, int],
            axis: str,
            axes_segmentations: Dict
    ) -> torch.Tensor:
        axes_boxes = axes_segmentations['boxes']

        if axis == 'y':
            tick_label_coords = torch.tensor([
                (x[1] + x[3]) / 2 for x in axes_boxes])
            dist = (tick_label_coords - coord[1]).abs()
        else:
            tick_label_coords = torch.tensor([
                (x[0] + x[2]) / 2 for x in axes_boxes])
            dist = (tick_label_coords - coord[0]).abs()

        return dist.argsort()[0].item()

    @staticmethod
    def _get_plot_values_in_img_coordinates(
        boxes: torch.Tensor,
        plot_type
    ):
        plot_vals = []
        boxes = boxes.numpy()
        for box in boxes:
            x1, y1, x2, y2 = box
            if plot_type == 'vertical_bar':
                plot_vals.append(((x1 + x2) / 2, y1))
            elif plot_type == 'horizontal_bar':
                plot_vals.append((x2, (y1 + y2) / 2))
            elif plot_type in ('scatter', 'dot'):
                plot_vals.append(((x2 + x1) / 2, (y2 + y1) / 2))
        return plot_vals

    @staticmethod
    def _get_line_plot_values_in_img_coordinates(
        axes_segmentations,
        line_plot_mask: torch.Tensor
    ):
        # resizing since segmentation model trained with 256x256, and
        # axes segmentation model trained with 448x448
        line_plot_mask = T.Resize([448, 448])(line_plot_mask)
        line_plot_mask = line_plot_mask.squeeze()

        plot_vals = []
        for mask in axes_segmentations['x-axis']['masks']:
            rect = DetectText.get_min_area_rect(mask=mask)
            box_points = cv2.boxPoints(rect)

            # fix for box points outside of image
            for i in range(box_points.shape[0]):
                box_points[i][0] = max(box_points[i][0], 0)
                box_points[i][0] = min(mask.shape[1]-1, box_points[i][0])

                box_points[i][1] = max(box_points[i][1], 0)
                box_points[i][1] = min(mask.shape[0]-1, box_points[i][1])

            angle = rect[-1]

            if abs(angle - 90) / 90 < .05 or abs(angle) < .05:
                # not a rotated tick label
                x_val = int((box_points[2][0] + box_points[3][0]) / 2)
            else:
                # rotated tick label
                midpoint = ((box_points[2][0] - box_points[1][0]) / 2)
                x_val = int(box_points[1][0] + midpoint)
            if (line_plot_mask[:, x_val] == 0).all():
                # mask is missing here. find nearest point
                mask_vals = torch.where(line_plot_mask)
                new_x_val = (
                    mask_vals[1][(mask_vals[1] - x_val).abs()
                                                       .argsort()[0]]
                    .item())
                if abs(new_x_val - x_val) > 10:
                    # there's really not a part of the line at that tick
                    continue
                else:
                    x_val = new_x_val
            y_val = (torch.argwhere(line_plot_mask[:, x_val])
                     .type(torch.float)
                     .mean()
                     .type(torch.int)
                     .item()
                     )
            plot_vals.append((x_val, y_val))
        return plot_vals

    @staticmethod
    def _get_tick_points(
        axes_segmentations
    ):
        axes = {
            'x-axis': [],
            'y-axis': []
        }
        tick_id = 0
        for axis in axes_segmentations:
            for mask in axes_segmentations[axis]['masks']:
                rect = DetectText.get_min_area_rect(mask=mask)
                box_points = cv2.boxPoints(rect)

                angle = rect[-1]

                if abs(angle - 90) / 90 < .05 or abs(angle) < .05:
                    # not a rotated tick label
                    center = rect[0]
                    if axis == 'x-axis':
                        tick_pt = center[0]
                    else:
                        tick_pt = center[1]
                else:
                    # rotated tick label
                    midpoint = ((box_points[2][0] - box_points[1][0]) / 2)
                    tick_pt = box_points[1][0] + midpoint

                axes[axis].append({
                    'tick_id': tick_id,
                    'tick_pt': tick_pt
                })
                tick_id += 1
        return axes

    def _detect_plot_values_with_model(
            self,
            plot_types,
            plot_ids,
            model: LightningModule
    ):
        data_module_kwargs = {
            'is_train': False
        }

        if isinstance(model, DetectPlotValuesModel):

            def transform(image_id):
                if plot_types[image_id] == 'horizontal_bar':
                    # Rotating horizontal bar since underrepresented in data
                    # model does better if bars vertical
                    return albumentations.Compose([
                        albumentations.Resize(height=448, width=448),
                        albumentations.Rotate(limit=(90, 90),
                                              p=1.0),
                        albumentations.Normalize(mean=0, std=1),
                        ToTensorV2()
                    ])
                else:
                    return albumentations.Compose([
                        albumentations.Resize(height=448, width=448),
                        albumentations.Normalize(mean=0, std=1),
                        ToTensorV2()
                    ])
        else:
            transform = albumentations.Compose([
                albumentations.Resize(height=256, width=256),
                albumentations.Normalize(
                    mean=FCN_ResNet50_Weights.DEFAULT.transforms().mean,
                    std=FCN_ResNet50_Weights.DEFAULT.transforms().std),
                ToTensorV2()
            ])

        def collate_func(batch):
            return tuple(zip(*batch))
        collate_func = collate_func if \
            isinstance(model, DetectPlotValuesModel) else None

        data_module = PlotDataModule(
            plots_dir=self.args['plots_dir'],
            batch_size=self.args['batch_size'],
            dataset_class=DetectPlotValuesDataset,
            num_workers=0 if self._is_debug else os.cpu_count(),
            file_id_chart_type_map_path=plot_types,
            plot_ids=plot_ids,
            inference_transform=transform,
            collate_func=collate_func,
            **data_module_kwargs
        )
        predictions = self._trainer.predict(
            model=model,
            datamodule=data_module
        )
        return predictions

    @staticmethod
    def _get_plot_bounding_boxes(axes_segmentations: Dict) -> Dict:
        bounding_boxes = {}
        for file_id in axes_segmentations:
            x_axis_segmentations = axes_segmentations[file_id]['x-axis']
            y_axis_segmentations = axes_segmentations[file_id]['y-axis']

            if len(x_axis_segmentations['boxes']) > 0:
                x0 = (x_axis_segmentations['boxes'][0][0] +
                      x_axis_segmentations['boxes'][0][2]) / 2
            else:
                x0 = torch.tensor(0)

            if len(y_axis_segmentations['boxes']) > 0:
                y0 = (y_axis_segmentations['boxes'][-1][1] +
                      y_axis_segmentations['boxes'][-1][3]) / 2
            else:
                y0 = torch.tensor(0)

            if len(x_axis_segmentations['boxes']) > 0:
                x1 = x_axis_segmentations['boxes'][-1][2]
            else:
                x1 = torch.tensor(448)

            if len(y_axis_segmentations['boxes']) > 0:
                y1 = (y_axis_segmentations['boxes'][0][3] +
                      y_axis_segmentations['boxes'][0][1]) / 2
            else:
                y1 = torch.tensor(448)

            if torch.has_cuda:
                x0 = x0.to('cuda')
                x1 = x1.to('cuda')
                y0 = y0.to('cuda')
                y1 = y1.to('cuda')

            height = y1 - y0
            width = x1 - x0
            bounding_boxes[file_id] = {
                'plot_bbox': {
                    'x0': int(x0.item()),
                    'y0': int(y0.item()),
                    'width': int(width.item()),
                    'height': int(height.item())
                }
            }
        return bounding_boxes

    def _classify_plot_type(self, axes_segmentations: Dict) -> Dict[str, str]:
        transforms = albumentations.Compose([
            albumentations.Resize(height=256, width=256),
            albumentations.CenterCrop(height=240, width=240),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        bounding_boxes = self._get_plot_bounding_boxes(
            axes_segmentations=axes_segmentations
        )
        plot_ids = list(axes_segmentations.keys())
        dataset = ClassifyPlotTypeDataset(
            plots_dir=self.args['plots_dir'],
            plot_ids=plot_ids,
            transform=transforms,
            plot_meta=bounding_boxes,
            is_train=False
        )
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.args['batch_size'],
            num_workers=os.cpu_count(),
            shuffle=False
        )
        predictions = self._trainer.predict(
            model=self._classify_plot_type_model,
            dataloaders=data_loader
        )
        predictions = np.concatenate(predictions)

        predictions = dict(zip(plot_ids, predictions))
        return predictions

    def _find_axes_tick_labels(self) -> Dict:
        batch_size = self.args['batch_size']
        for start in torch.arange(0, len(self._plot_ids), batch_size):
            data_module = PlotDataModule(
                plots_dir=self.args['plots_dir'],
                batch_size=batch_size,
                dataset_class=FindAxesTickLabelsDataset,
                num_workers=0 if self._is_debug else os.cpu_count(),
                plot_ids=self._plot_ids[start:start+batch_size],
                inference_transform=albumentations.Compose([
                    albumentations.Resize(height=448, width=448),
                    albumentations.Normalize(mean=0, std=1),
                    ToTensorV2()
                ]),
                is_train=False
            )

            predictions = self._trainer.predict(
                model=self._detect_axes_labels_model,
                datamodule=data_module
            )
            yield predictions[0]

    def _find_axes_tick_labels_for_horizontal_bar(self, plot_ids) -> Dict:
        """
        "Fixes" horizontal bar axes labels by rotating vertically since
        vertical bar much more represented in the data
        We then rotate boxes/masks back to original

        :param plot_ids:
        :return:
        """
        data_module = PlotDataModule(
            plots_dir=self.args['plots_dir'],
            batch_size=self.args['batch_size'],
            dataset_class=FindAxesTickLabelsDataset,
            num_workers=0 if self._is_debug else os.cpu_count(),
            plot_ids=plot_ids,
            inference_transform=albumentations.Compose([
                albumentations.Resize(height=448, width=448),
                albumentations.Rotate(
                    limit=(90, 90),
                    p=1.0),
                albumentations.Normalize(mean=0, std=1),
                ToTensorV2()
            ]),
            is_train=False
        )
        predictions = self._trainer.predict(
            model=self._detect_axes_labels_model,
            datamodule=data_module
        )
        predictions = predictions[0]

        d = {}

        # rotate preds back
        for image_id in predictions:
            rotation_transform = albumentations.Compose(
                [
                    albumentations.Rotate(limit=(-90, -90),
                                          p=1.0)
                ],
                bbox_params=albumentations.BboxParams(
                    format='pascal_voc',
                    label_fields=['class_labels']
                )
            )

            if predictions[image_id]['x-axis']['masks'].shape[0] > 0:
                transformed = rotation_transform(
                    image=np.zeros((448, 448)),
                    bboxes=predictions[image_id]['x-axis']['boxes'],
                    masks=[x.numpy() for x in predictions[image_id]['x-axis']['masks']],
                    class_labels=predictions[image_id]['x-axis']['labels']
                )
                transformed['masks'] = torch.tensor(
                    np.array(transformed['masks'])).to(
                        predictions[image_id]['y-axis']['masks'].device)
                transformed['bboxes'] = torch.tensor(
                    np.array(transformed['bboxes'])).to(
                        predictions[image_id]['y-axis']['boxes'].device)

                sort_idx = sort_boxes(
                    transformed['bboxes'],
                    axis='y-axis')
                transformed['bboxes'] = transformed['bboxes'][sort_idx]
                transformed['masks'] = transformed['masks'][sort_idx]

            else:
                transformed = None
            d[image_id] = transformed
        return d

    def _detect_axes_label_text(
        self,
        axes_segmentations: Dict,
        plot_types: Dict[str, str]
    ):
        reader = easyocr.Reader(
            lang_list=['en'],
            recog_network='ocr_model',
            user_network_directory=self.args['ocr_user_network_directory'],
            model_storage_directory=self.args['ocr_model_storage_directory'],
            gpu=torch.cuda.is_available()
        )
        dt = DetectText(
            axes_segmentations=axes_segmentations,
            images_dir=Path(self.args['plots_dir']),
            plot_types=plot_types,
            easyocr_reader=reader
        )
        return dt.run_inference()

    @staticmethod
    def _construct_data_series(
            plot_types: Dict[str, str],
            file_id_plot_values_map: Dict[str, List[Tuple[Any, Any]]]):
        res = []
        for file_id, plot_values in file_id_plot_values_map.items():
            for axis, axis_idx in [('x', 0), ('y', 1)]:
                data_series = ';'.join([
                    str(x[axis_idx]) for x in plot_values
                    if x[axis_idx] != 'HISTOGRAM_PLACEHOLDER'])
                res.append({
                    'id': f'{file_id}_{axis}',
                    'data_series': data_series,
                    'chart_type': plot_types[file_id]
                })
        return res
