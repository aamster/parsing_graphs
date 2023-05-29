import json
import logging
import os
from pathlib import Path
from typing import Dict

import argschema as argschema
import numpy as np
import pandas as pd
import torch
from lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b1
import torchvision.transforms.v2 as T
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, \
    maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import InterpolationMode

from parse_plots.classify_plot_type.dataset import ClassifyPlotTypeDataset
from parse_plots.classify_plot_type.model import ClassifyPlotTypeModel
from parse_plots.detect_axes_labels_text.detect_text import DetectText
from parse_plots.find_axes_tick_labels.dataset import FindAxesTickLabelsDataset
from parse_plots.find_axes_tick_labels.model import SegmentAxesTickLabelsModel


class ParsePlotsSchema(argschema.ArgSchema):
    plots_dir = argschema.fields.InputDir(required=True)
    annotations_dir = argschema.fields.InputDir(required=True)
    classify_plot_type_checkpoint = argschema.fields.InputFile(required=True)
    segment_axes_tick_labels_checkpoint = argschema.fields.InputFile(required=True)
    out_dir = argschema.fields.OutputDir(required=True)
    is_debug = argschema.fields.Boolean(default=False)


class ParsePlotsRunner(argschema.ArgSchemaParser):
    default_schema = ParsePlotsSchema

    def __init__(self):
        super().__init__(
            schema_type=ParsePlotsSchema
        )
        plot_files = os.listdir(self.args['plots_dir'])
        self._plot_ids = [Path(x).stem for x in plot_files]

    def run(self):
        self.logger.info('Classifying plot type')
        plot_types = self._classify_plot_type()

        self.logger.info('segmenting axes labels')
        axes_segmentations = self._find_axes_tick_labels()

        self.logger.info('detecting axes label text')
        tick_labels = self._detect_axes_label_text(
            axes_segmentations=axes_segmentations)

        out_path = Path(self.args['out_dir']) / 'submission.csv'
        pd.DataFrame(self._construct_data_series(
            plot_types=plot_types,
            tick_labels=tick_labels
        )).to_csv(out_path, index=False)
        self.logger.info(f'Wrote submission to {out_path}')

    def _classify_plot_type(self):
        transforms = T.Compose([
            T.ToTensor(),
            T.Resize(
                size=(256, 256),
                interpolation=InterpolationMode.BICUBIC
            ),
            T.CenterCrop(size=(240, 240)),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        dataset = ClassifyPlotTypeDataset(
            annotations_dir=self.args['annotations_dir'],
            plots_dir=self.args['plots_dir'],
            plot_ids=self._plot_ids,
            transform=transforms
        )
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=1 if self.args['is_debug'] else 64,
            num_workers=os.cpu_count(),
            shuffle=False
        )
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

        trainer = Trainer(
            limit_predict_batches=2 if self.args['is_debug'] else None)
        predictions = trainer.predict(
            model=model,
            dataloaders=data_loader
        )
        predictions = np.concatenate(predictions)

        predictions = dict(zip(self._plot_ids, predictions))
        return predictions

    def _find_axes_tick_labels(self):
        transforms = T.Compose([
            T.ToImageTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Resize([448, 448]),
            T.SanitizeBoundingBox(labels_getter=lambda inputs: inputs[3])
        ])

        dataset = FindAxesTickLabelsDataset(
            annotations_dir=self.args['annotations_dir'],
            plots_dir=self.args['plots_dir'],
            plot_ids=self._plot_ids,
            transform=transforms
        )
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=1 if self.args['is_debug'] else 64,
            num_workers=0 if self.args['is_debug'] else os.cpu_count(),
            shuffle=False
        )

        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn_v2(weights=weights)

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
                checkpoint_path=self.args['segment_axes_tick_labels_checkpoint'],
                learning_rate=1e-3,
                model=model,
                hyperparams={},
                map_location=(torch.device('cpu') if not torch.has_cuda
                              else None)
        )
        trainer = Trainer(
            limit_predict_batches=2 if self.args['is_debug'] else None
        )
        predictions = trainer.predict(model=model, dataloaders=data_loader)
        predictions = np.concatenate(predictions)

        res = {}
        for i in range(len(predictions)):
            res[self._plot_ids[i]] = predictions[i]

        return res

    def _detect_axes_label_text(
        self,
        axes_segmentations: Dict
    ):
        dt = DetectText(
            axes_segmentations=axes_segmentations,
            images_dir=Path(self.args['plots_dir']))
        return dt.run()

    @staticmethod
    def _construct_data_series(
            plot_types: Dict[str, str],
            tick_labels):
        res = []
        for file_id, plot_type in plot_types.items():
            for axis, axis_labels in tick_labels[file_id].items():
                axis = 'x' if axis == 'x-axis' else 'y'
                if axis == 'x':     # TODO remove when y axis values complete
                    data_series = ';'.join([str(x) for x in axis_labels])
                    res.append({
                        'id': f'{file_id}_{axis}',
                        'data_series': data_series,
                        'chart_type': plot_type
                    })
        return res


if __name__ == '__main__':
    ParsePlotsRunner().run()
