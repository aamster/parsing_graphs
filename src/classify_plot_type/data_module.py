import os
from pathlib import Path

import lightning
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, InterpolationMode

from classify_plot_type.dataset import PlotDataset


class PlotDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        plots_dir,
        annotations_dir
    ):
        super().__init__()
        self._batch_size = batch_size
        self._plots_dir = plots_dir
        self._annotations_dir = annotations_dir
        self._train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                size=(256, 256),
                interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop(size=(240, 240)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        self._test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                size=(256, 256),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(size=(240, 240)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        self._train = None
        self._val = None
        self._num_workers = num_workers

    def setup(self, stage: str):
        if stage == "fit":
            plot_files = os.listdir(self._plots_dir)
            plot_ids = np.array([Path(x).stem for x in plot_files])
            idxs = np.arange(len(plot_ids))
            rng = np.random.default_rng(1234)
            rng.shuffle(idxs)
            train_idxs = idxs[:int(len(idxs) * .7)]
            val_idxs = idxs[int(len(idxs) * .7):]

            self._train = PlotDataset(
                plots_dir=self._plots_dir,
                annotations_dir=self._annotations_dir,
                plot_ids=plot_ids[train_idxs],
                transform=self._train_transform
            )
            self._val = PlotDataset(
                plots_dir=self._plots_dir,
                annotations_dir=self._annotations_dir,
                plot_ids=plot_ids[val_idxs],
                transform=self._test_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self._train,
            batch_size=self._batch_size,
            num_workers=self._num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self._val,
            batch_size=self._batch_size,
            num_workers=self._num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self._val,
            batch_size=self._batch_size,
            num_workers=self._num_workers
        )
