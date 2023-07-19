from contextlib import contextmanager
from typing import Dict, Any

import lightning
import numpy as np
import torch.nn
from torchmetrics.detection import MeanAveragePrecision

from parse_plots.detect_axes_labels_text.detect_text import sort_boxes
from parse_plots.utils import threshold_soft_masks, convert_to_tensor


@contextmanager
def evaluate(model: torch.nn.Module):
    """Temporarily switch to evaluation mode."""
    istrain = model.training
    try:
        model.eval()
        yield model
    finally:
        if istrain:
            model.train()


class SegmentAxesTickLabelsModel(lightning.LightningModule):
    def __init__(
        self,
        learning_rate,
        model: torch.nn.Module,
        hyperparams: Dict,
    ):
        super().__init__()
        self.model = model

        self.train_map = MeanAveragePrecision(
            iou_type='segm',
            max_detection_thresholds=[100]
        )
        self.val_map = MeanAveragePrecision(
            iou_type='segm',
            max_detection_thresholds=[100]
        )
        self._learning_rate = learning_rate
        self._hyperparams = hyperparams

    def training_step(self, batch, batch_idx):
        data, target = batch
        losses = self.model(data, target)

        # sums the classification and regression losses for both the
        # RPN and the R-CNN, and the mask loss
        loss = sum(loss for loss in losses.values())
        metrics = {
            f'train_{loss_name}': loss
            for loss_name, loss in losses.items()
        }
        metrics['train_loss'] = loss
        self.log_dict(metrics)

        preds = self._get_predictions(batch=batch)
        target = convert_to_tensor(target=target)
        self.train_map.update(preds=preds, target=target)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self._get_predictions(batch=batch)
        target = convert_to_tensor(target=target)
        self.val_map.update(preds=preds, target=target)

    def predict_step(
            self,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0
    ):
        img, target = batch
        preds = self._get_predictions(batch=batch)

        sorted_preds = {}
        for pred_idx in range(len(preds)):
            # only include confident predictions
            preds[pred_idx]['boxes'] = preds[pred_idx]['boxes'][preds[pred_idx]['scores'] > 0.5]
            preds[pred_idx]['masks'] = preds[pred_idx]['masks'][preds[pred_idx]['scores'] > 0.5]
            preds[pred_idx]['labels'] = preds[pred_idx]['labels'][preds[pred_idx]['scores'] > 0.5]

            preds[pred_idx] = self._remove_outlier_box_predictions(
                pred=preds[pred_idx]
            )

            boxes = preds[pred_idx]['boxes']
            masks = preds[pred_idx]['masks']
            labels = preds[pred_idx]['labels']

            x_axis_idxs = torch.where(labels == 1)[0]
            y_axis_idxs = torch.where(labels == 2)[0]
            x_axis_sort_idx = sort_boxes(
                boxes[x_axis_idxs],
                axis='x-axis')
            y_axis_sort_idx = sort_boxes(
                boxes[y_axis_idxs],
                axis='y-axis')
            sorted_preds[target['image_id'][pred_idx]] = {
                'x-axis': {
                    'boxes': boxes[x_axis_idxs][x_axis_sort_idx],
                    'masks': masks[x_axis_idxs][x_axis_sort_idx],
                    'labels': labels[x_axis_idxs][x_axis_sort_idx],
                },
                'y-axis': {
                    'boxes': boxes[y_axis_idxs][y_axis_sort_idx],
                    'masks': masks[y_axis_idxs][y_axis_sort_idx],
                    'labels': labels[y_axis_idxs][y_axis_sort_idx],
                }
            }

        return sorted_preds

    def _get_predictions(self, batch):
        with evaluate(model=self.model):
            with torch.no_grad():
                data, _ = batch
                preds = self.model(data)

        preds = threshold_soft_masks(preds=preds)

        return preds

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(params=self._hyperparams)

    def on_train_epoch_end(self) -> None:
        self.log_dict({
            'train_map': self.train_map.compute()['map'].item()
        })
        self.train_map.reset()

    def on_validation_epoch_end(self) -> None:
        map = self.val_map.compute()['map'].item()
        self.log_dict({
            'val_map': map
        })
        self.val_map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        return optimizer

    def _remove_outlier_box_predictions(self, pred: Dict):
        """Sometimes there are stray boxes that are not axes labels.
        This tries to remove those"""
        label_int_str_map = {
            1: 'x-axis',
            2: 'y-axis'
        }
        preds = {
            'boxes': [],
            'masks': [],
            'labels': []
        }

        def does_box_overlap(box, other_boxes, axis):
            box_x1, box_y1, box_x2, box_y2 = box
            if axis == 'x-axis':
                box1, box2 = [box_y1, box_y2]
                other1, other2 = [
                    torch.quantile(other_boxes[:, 1], 0.5).item(),
                    torch.quantile(other_boxes[:, 3], 0.5).item()
                ]
            else:
                box1, box2 = [box_x1, box_x2]
                other1, other2 = [
                    torch.quantile(other_boxes[:, 0], 0.5).item(),
                    torch.quantile(other_boxes[:, 2], 0.5).item()
                ]

            #   ---------
            # --------------
            if box1 >= other1 and box2 <= other2:
                return True

            # --------------
            #   ---------
            elif box1 <= other1 and box2 >= other2:
                return True

            # --------------
            #           ---------
            elif box1 <= other1 <= box2 <= other2:
                return True

            #            ---------
            # --------------
            elif other1 <= box1 <= other2:
                return True
            else:
                return False

        for label in (1, 2):
            label_idx = torch.where(pred['labels'] == label)[0]
            boxes = pred['boxes'][label_idx]
            overlap = [
                does_box_overlap(
                    box=boxes[i],
                    other_boxes=boxes[[idx for idx in range(len(boxes)) if idx != i]],
                    axis=label_int_str_map[label]
                ) for i in range(len(boxes))
            ]

            preds['boxes'].append(boxes[overlap])
            preds['masks'].append(pred['masks'][label_idx][overlap])
            preds['labels'].append(pred['labels'][label_idx][overlap])
        preds['boxes'] = torch.concatenate(preds['boxes'])
        preds['masks'] = torch.concatenate(preds['masks'])
        preds['labels'] = torch.concatenate(preds['labels'])

        return preds
