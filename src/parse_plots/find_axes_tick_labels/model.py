from contextlib import contextmanager
from typing import Dict, Any

import lightning
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
        hyperparams: Dict
    ):
        super().__init__()
        self.model = model

        self.train_map = MeanAveragePrecision(
            iou_type='segm',
            max_detection_thresholds=[50]
        )
        self.val_map = MeanAveragePrecision(
            iou_type='segm',
            max_detection_thresholds=[50]
        )
        self._learning_rate = learning_rate
        self._hyperparams = hyperparams

    def training_step(self, batch, batch_idx):
        data, target = batch
        losses = self.model(data, target)

        # sums the classification and regression losses for both the
        # RPN and the R-CNN, and the mask loss
        loss = sum(loss for loss in losses.values())

        self.log("train_loss", loss, prog_bar=True)
        metrics = {
            f'train_{loss_name}': loss
            for loss_name, loss in losses.items()
        }
        metrics['train_loss'] = loss
        self.logger.log_metrics(metrics, step=self.global_step)

        preds = self._get_predictions(batch=batch)
        self.train_map.update(preds=preds, target=target)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self.model(data)
        self.val_map.update(preds=preds, target=target)

    def predict_step(
            self,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0
    ):
        preds = self._get_predictions(batch=batch)
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
        self.logger.log_metrics({
            'train_map': self.train_map.compute()['map'].item()
        }, step=self.current_epoch)
        self.train_map.reset()

    def on_validation_epoch_end(self) -> None:
        map = self.val_map.compute()['map'].item()
        self.log('val_map', map, on_epoch=True)
        self.logger.log_metrics({
            'val_map': map
        }, step=self.current_epoch)
        self.val_map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        return optimizer