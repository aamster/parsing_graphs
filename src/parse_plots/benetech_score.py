import argparse
import json
from pathlib import Path

import pandas as pd

from parse_plots.metrics import benetech_score

parser = argparse.ArgumentParser()
parser.add_argument('--preds_path', required=True)
parser.add_argument('--annotations_dir', required=True)
args = parser.parse_args()


def main():
    preds = pd.read_csv(args.preds_path)
    preds['data_series'] = preds['data_series'].str.split(';')

    ground_truth = []
    pred_ids = set(preds['id'].apply(lambda x: x.split('_')[0]))

    for file_id in pred_ids:
        with open(Path(args.annotations_dir) / f'{file_id}.json') as f:
            a = json.load(f)
            ground_truth.append({
                'id': f'{file_id}_x',
                'data_series': [x['x'] for x in a['data-series']],
                'chart_type': a['chart-type']
            })
            ground_truth.append({
                'id': f'{file_id}_y',
                'data_series': [x['y'] for x in a['data-series']],
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
