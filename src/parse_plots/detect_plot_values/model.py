from contextlib import contextmanager
from typing import Dict, Any

import lightning
import torch.nn
from torchmetrics.detection import MeanAveragePrecision

from parse_plots.utils import threshold_soft_masks, convert_masks_to_tensor


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


class DetectPlotValuesModel(lightning.LightningModule):
    def __init__(
        self,
        learning_rate,
        model: torch.nn.Module,
        hyperparams: Dict,
    ):
        super().__init__()
        self.model = model

        self._learning_rate = learning_rate
        self._hyperparams = hyperparams
        self.train_map = MeanAveragePrecision(
            iou_type='bbox',
            max_detection_thresholds=[200],
            iou_thresholds=[0.5]
        )
        self.val_map = MeanAveragePrecision(
            iou_type='bbox',
            max_detection_thresholds=[200],
            iou_thresholds=[0.5]
        )

    def training_step(self, batch, batch_idx):
        data, target = batch
        losses = self.model(data, target)

        # sums the classification and regression losses for both the
        # RPN and the R-CNN
        loss = sum(loss for loss in losses.values())
        metrics = {
            f'train_{loss_name}': loss
            for loss_name, loss in losses.items()
        }
        metrics['train_loss'] = loss
        self.log_dict(metrics)

        preds = self._get_predictions(batch=batch)

        self.train_map.update(preds=preds, target=target)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self._get_predictions(batch=batch)
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
            preds[i]['labels'] = preds[i]['labels'][preds[i]['scores'] > 0.5]

        return preds

    def _get_predictions(self, batch):
        with evaluate(model=self.model):
            with torch.no_grad():
                data, target = batch
                preds = self.model(data)

        return preds

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(params=self._hyperparams)

    def on_train_epoch_end(self) -> None:
        self.log_dict({
            'train_map': self.train_map.compute()['map'].item()
        })
        self.train_map.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict({
            'val_map': self.val_map.compute()['map'].item()
        })
        self.val_map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        return optimizer
