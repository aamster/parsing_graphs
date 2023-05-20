import os
from pathlib import Path
from typing import Optional, Union, Type, Callable

import lightning
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from parse_plots.classify_plot_type.dataset import PlotDataset \
    as ClassifyPlotTypesDataset
from parse_plots.find_axes_tick_labels.dataset import PlotDataset \
    as AxesTickLabelsDataset


class PlotDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        plots_dir,
        annotations_dir,
        dataset_class: Union[Type[ClassifyPlotTypesDataset],
                             Type[AxesTickLabelsDataset]],
        train_transform: Optional[transforms.Compose] = None,
        inference_transform: Optional[transforms.Compose] = None,
        collate_func: Optional[Callable] = None
    ):
        super().__init__()
        self._batch_size = batch_size
        self._plots_dir = plots_dir
        self._annotations_dir = annotations_dir
        self._train_transform = train_transform
        self._test_transform = inference_transform
        self._train = None
        self._val = None
        self._num_workers = num_workers
        self._dataset_class = dataset_class
        self._collate_func = collate_func

    def setup(self, stage: str):
        if stage == "fit":
            plot_files = os.listdir(self._plots_dir)
            plot_ids = np.array([Path(x).stem for x in plot_files])
            idxs = np.arange(len(plot_ids))
            rng = np.random.default_rng(1234)
            rng.shuffle(idxs)
            train_idxs = idxs[:int(len(idxs) * .7)]
            val_idxs = idxs[int(len(idxs) * .7):]

            self._train = self._dataset_class(
                plots_dir=self._plots_dir,
                annotations_dir=self._annotations_dir,
                plot_ids=plot_ids[train_idxs],
                transform=self._train_transform
            )
            self._val = self._dataset_class(
                plots_dir=self._plots_dir,
                annotations_dir=self._annotations_dir,
                plot_ids=plot_ids[val_idxs],
                transform=self._test_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self._train,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=self._collate_func
        )

    def val_dataloader(self):
        return DataLoader(
            self._val,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=self._collate_func
        )

    def predict_dataloader(self):
        return DataLoader(
            self._val,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=self._collate_func
        )