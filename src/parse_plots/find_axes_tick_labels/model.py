from contextlib import contextmanager
from typing import Dict, Any, List

import lightning
import numpy as np
import torch.nn
from torchmetrics.detection import MeanAveragePrecision


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
        box_outlier_threshold: float = 1.5
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
        self._box_outlier_threshold = box_outlier_threshold

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
        target = self._convert_masks_to_tensor(target=target)
        self.train_map.update(preds=preds, target=target)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self._get_predictions(batch=batch)
        target = self._convert_masks_to_tensor(target=target)
        self.val_map.update(preds=preds, target=target)

    def predict_step(
            self,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0
    ):
        preds = self._get_predictions(batch=batch)

        for i in range(len(preds)):
            # only include confident predictions
            preds[i]['boxes'] = preds[i]['boxes'][preds[i]['scores'] > 0.5]
            preds[i]['masks'] = preds[i]['masks'][preds[i]['scores'] > 0.5]
            preds[i]['labels'] = preds[i]['labels'][preds[i]['scores'] > 0.5]

            preds[i] = self._remove_outlier_box_predictions(
                pred=preds[i]
            )

        return preds

    def _get_predictions(self, batch):
        with evaluate(model=self.model):
            with torch.no_grad():
                data, target = batch
                preds = self.model(data)

        # threshold soft masks
        for pred_idx, pred in enumerate(preds):
            masks = torch.zeros_like(preds[pred_idx]['masks'],
                                     dtype=torch.uint8)
            for mask_idx, mask in enumerate(pred['masks']):
                mask = (mask > 0.5).type(torch.uint8)
                masks[mask_idx] = mask
            preds[pred_idx]['masks'] = masks
            if len(preds[pred_idx]['masks'].shape) == 4:
                preds[pred_idx]['masks'] = preds[pred_idx]['masks'].squeeze(
                    dim=1)

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

    @staticmethod
    def _convert_masks_to_tensor(target):
        for i in range(len(target)):
            if type(target[i]['masks']) is not torch.Tensor:
                target[i]['masks'] = target[i]['masks'].data
        return target

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

        def get_box_center(box):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            return x1 + int(w / 2), y1 + int(h / 2)

        for label in (1, 2):
            label_idx = np.where(pred['labels'] == label)[0]
            boxes = pred['boxes'][label_idx]
            centers = [get_box_center(box=box) for box in boxes]

            if label_int_str_map[label] == 'x-axis':
                center = np.array([c[1] for c in centers])
            else:
                center = np.array([c[0] for c in centers])

            Q1, Q3 = np.quantile(center, (0.25, 0.75))
            IQR = Q3 - Q1
            lower = Q1 - self._box_outlier_threshold * IQR
            upper = Q3 + self._box_outlier_threshold * IQR

            preds['boxes'].append(boxes[
                                      (center >= lower) &
                                      (center <= upper)])
            preds['masks'].append(pred['masks'][label_idx][
                                      (center >= lower) &
                                      (center <= upper)])
            preds['labels'].append(pred['labels'][label_idx][
                                      (center >= lower) &
                                      (center <= upper)])
        preds['boxes'] = torch.concatenate(preds['boxes'])
        preds['masks'] = torch.concatenate(preds['masks'])
        preds['labels'] = torch.concatenate(preds['labels'])

        return preds
