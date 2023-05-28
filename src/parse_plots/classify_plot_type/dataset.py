import json
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch.utils.data
from torch.utils.data.dataset import T_co

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
        annotations_dir,
        transform
    ):
        super().__init__()
        plot_ids = set(plot_ids)
        plot_files = os.listdir(plots_dir)
        self._plot_files = [x for x in plot_files if Path(x).stem in plot_ids][:2]
        self._plots_dir = Path(plots_dir)
        self._annotations_dir = Path(annotations_dir)
        self._transform = transform

    def __getitem__(self, index) -> T_co:
        id = Path(self._plot_files[index]).stem
        data = plt.imread(self._plots_dir / f'{id}.jpg')
        with open(self._annotations_dir / f'{id}.json') as f:
            a = json.load(f)

        plot_bb = a['plot-bb']

        # Correct for negative positions
        y0 = max(0, plot_bb['y0'])
        x0 = max(0, plot_bb['x0'])

        # Limit to the plot bounding box to exclude title/axes, etc
        data = data.copy()
        data = data[
               y0:y0 + plot_bb['height'],
               x0:x0 + plot_bb['width']]
        target = plot_type_id_map[a['chart-type']]

        data = self._transform(data)

        return data, target

    def __len__(self):
        return len(self._plot_files)
