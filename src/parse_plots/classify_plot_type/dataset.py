import json
import os
from pathlib import Path
from typing import List, Optional, Dict

import cv2
import torch.utils.data
from torch.utils.data.dataset import T_co
from torchvision.io import read_image

from parse_plots.utils import resize_plot_bounding_box

plot_type_id_map = {
    'dot': 0,
    'horizontal_bar': 1,
    'vertical_bar': 2,
    'line': 3,
    'scatter': 4
}
plot_id_type_map = {
    0: 'dot',
    1: 'horizontal_bar',
    2: 'vertical_bar',
    3: 'line',
    4: 'scatter'
}


class ClassifyPlotTypeDataset(torch.utils.data.Dataset):
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
        plot_ids = set(plot_ids)
        plot_files = os.listdir(plots_dir)
        self._plot_files = [x for x in plot_files if Path(x).stem in plot_ids]
        self._plots_dir = Path(plots_dir)
        self._annotations_dir = Path(annotations_dir) \
            if annotations_dir is not None else None
        self._transform = transform
        self._is_train = is_train
        if is_train and annotations_dir is None:
            raise ValueError('if train, annotations_dir must be given')
        if not is_train:
            if plot_meta is None:
                raise ValueError('if test, plot_id_meta must be given')
        self._plot_meta = plot_meta

    def __getitem__(self, index) -> T_co:
        id = Path(self._plot_files[index]).stem
        img = cv2.imread(str(self._plots_dir / f'{id}.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self._is_train:
            with open(self._annotations_dir / f'{id}.json') as f:
                a = json.load(f)
            plot_bb = a['plot-bb']
            target = plot_type_id_map[a['chart-type']]

        else:
            plot_bb = self._plot_meta[id]['plot_bbox']
            plot_bb = resize_plot_bounding_box(
                img=img,
                plot_bounding_box=plot_bb
            )
            target = ''

        # Correct for negative positions
        y0 = max(0, plot_bb['y0'])
        x0 = max(0, plot_bb['x0'])

        # Limit to the plot bounding box to exclude title/axes, etc
        img = img[
               y0:y0 + plot_bb['height'],
               x0:x0 + plot_bb['width']]

        #######
        # DEBUG
        return img, target
        #######

        img = self._transform(image=img)['image']

        return img, target

    def __len__(self):
        return len(self._plot_files)
