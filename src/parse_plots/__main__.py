import json
import os
from pathlib import Path

import argschema as argschema
import numpy as np
import torch
from lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b1
from torchvision.transforms import transforms as T, InterpolationMode

from parse_plots.classify_plot_type.dataset import ClassifyPlotTypeDataset
from parse_plots.classify_plot_type.model import ClassifyPlotTypeModel


class ParsePlotsSchema(argschema.ArgSchema):
    plots_dir = argschema.fields.InputDir(required=True)
    annotations_dir = argschema.fields.InputDir(required=True)
    classify_plot_type_checkpoint = argschema.fields.InputFile(required=True)


class ParsePlotsRunner(argschema.ArgSchemaParser):
    default_schema = ParsePlotsSchema

    def __init__(self):
        super().__init__(
            schema_type=ParsePlotsSchema
        )
        plot_files = os.listdir(self.args['plots_dir'])
        self._plot_ids = [Path(x).stem for x in plot_files]

    def run(self):
        self._classify_plot_type()

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
            batch_size=64,
            num_workers=os.cpu_count()
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

        trainer = Trainer()
        predictions = trainer.predict(model=model, dataloaders=data_loader)
        predictions = np.concatenate(predictions)
        with open('plot_type_predictions.json', 'w') as f:
            f.write(json.dumps(dict(zip(self._plot_ids, predictions)),
                               indent=2))


if __name__ == '__main__':
    ParsePlotsRunner().run()
