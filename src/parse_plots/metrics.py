import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from rapidfuzz.distance.Levenshtein import distance as levenshtein


def sigmoid(x):
    return 2 - 2 / (1 + np.exp(-x))


def normalized_rmse(y_true, y_pred):
    # The argument to the sigmoid transform is equal to
    # rmse(y_true, y_pred) / rmse(y_true, np.mean(y_true))
    return sigmoid((1 - r2_score(y_true, y_pred)) ** 0.5)


def normalized_levenshtein_score(y_true, y_pred):
    total_distance = np.sum(
        [levenshtein(yt, yp) for yt, yp in zip(y_true, y_pred)])
    length_sum = np.sum([len(yt) for yt in y_true])
    return sigmoid(total_distance / length_sum)


def score_series(y_true, y_pred):
    if len(y_true) != len(y_pred):
        return 0.0
    if isinstance(y_true[0], str):
        y_pred = [str(x) for x in y_pred]
        return normalized_levenshtein_score(y_true, y_pred)
    else:
        y_pred = [float(x) for x in y_pred]
        return normalized_rmse(y_true, y_pred)


def benetech_score(ground_truth: pd.DataFrame,
                   predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate predictions using the metric from the Benetech - Making
    Graphs Accessible.

    Parameters
    ----------
    ground_truth: pd.DataFrame
        Has columns `[data_series, chart_type]` and an index `id`. Values in
        `data_series`
        should be either arrays of floats or arrays of strings.

    predictions: pd.DataFrame
    """
    if not ground_truth.index.equals(predictions.index):
        raise ValueError(
            "Must have exactly one prediction for each ground-truth instance.")
    if not ground_truth.columns.equals(predictions.columns):
        raise ValueError(
            f"Predictions must have columns: {ground_truth.columns}.")
    pairs = zip(ground_truth.itertuples(),
                predictions.itertuples(index=False))
    scores = []
    for (image_id, gt_series, gt_type), (pred_series, pred_type) in pairs:
        if gt_type != pred_type:  # Check chart_type condition
            scores.append({'image_id': image_id, 'score': 0.0})
        else:  # Score with RMSE or Levenshtein as appropriate
            score = score_series(gt_series, pred_series)
            scores.append({'image_id': image_id, 'score': score})
    scores = pd.DataFrame(scores).set_index('image_id')
    return scores
