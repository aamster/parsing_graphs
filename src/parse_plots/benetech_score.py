import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from parse_plots.metrics import benetech_score

parser = argparse.ArgumentParser()
parser.add_argument('--preds_path', required=True)
parser.add_argument('--annotations_dir', required=True)
args = parser.parse_args()


def main():
    preds = pd.read_csv(args.preds_path)
    preds['data_series'][preds['data_series'].isna()] = '0'
    preds['data_series'] = preds['data_series'].str.split(';')

    ground_truth = []
    pred_ids = set(preds['id'].apply(lambda x: x.split('_')[0]))

    for file_id in pred_ids:
        with open(Path(args.annotations_dir) / f'{file_id}.json') as f:
            a = json.load(f)
            data_series_x = []
            data_series_y = []
            for v in a['data-series']:
                x, y = v['x'], v['y']
                if a['chart-type'] == 'horizontal_bar':
                    if type(y) is not str:
                        x, y = y, x
                if type(x) is float:
                    if not np.isnan(x):
                        data_series_x.append(x)
                else:
                    data_series_x.append(x)
                if type(y) is float:
                    if not np.isnan(y):
                        data_series_y.append(y)
                else:
                    data_series_y.append(y)
            ground_truth.append({
                'id': f'{file_id}_x',
                'data_series': data_series_x,
                'chart_type': a['chart-type']
            })
            ground_truth.append({
                'id': f'{file_id}_y',
                'data_series': data_series_y,
                'chart_type': a['chart-type']
            })
    ground_truth = pd.DataFrame(ground_truth)

    preds = preds.set_index('id')
    ground_truth = ground_truth.set_index('id')

    preds = preds.sort_index()
    ground_truth = ground_truth.sort_index()

    score = benetech_score(
        ground_truth=ground_truth,
        predictions=preds
    )
    score = score.merge(ground_truth['chart_type'], left_index=True,
                        right_index=True)
    print(score.groupby('chart_type')['score'].mean())
    print(score['score'].mean())


if __name__ == '__main__':
    main()
