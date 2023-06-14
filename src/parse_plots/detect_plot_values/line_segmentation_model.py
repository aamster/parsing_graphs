from typing import Dict, Any

import lightning
import torch.nn
import torch.nn.functional as F
from torchmetrics import Dice


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (
                pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(
            dim=2) + smooth)))

    return loss.mean()


class SegmentLinePlotModel(lightning.LightningModule):
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
        self.train_dice = Dice(ignore_index=0)
        self.val_dice = Dice(ignore_index=0)

    def training_step(self, batch, batch_idx):
        data, target = batch

        preds = self.model(data)['out']
        loss, bce, dice = self._calc_loss(
            pred=preds,
            target=target['mask']
        )

        metrics = {
            'train_loss': loss,
            'train_bce_loss': bce,
            'train_dice_loss': dice
        }
        self.log_dict(metrics)

        self.train_dice.update(preds=preds, target=target['mask'])
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch

        with torch.no_grad():
            preds = self.model(data)['out']
        self.val_dice.update(preds=preds, target=target['mask'])

    def predict_step(
            self,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0
    ):
        data, target = batch
        with torch.no_grad():
            preds: torch.Tensor = self.model(data)['out']
        preds = (preds > 0.5).type(torch.uint8)

        return preds

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(params=self._hyperparams)

    def on_train_epoch_end(self) -> None:
        self.log_dict({
            'train_dice': self.train_dice.compute().item()
        })
        self.train_dice.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict({
            'val_dice': self.val_dice.compute().item()
        })
        self.val_dice.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        return optimizer

    @staticmethod
    def _calc_loss(pred, target, ce_weight=0.5):
        cross_entropy = F.cross_entropy(pred, target.long())

        pred = pred[:, 1]
        pred = F.sigmoid(pred)
        pred = pred.unsqueeze(dim=1)
        target = target.unsqueeze(dim=1)
        dice = dice_loss(pred, target)

        loss = cross_entropy * ce_weight + dice * (1 - ce_weight)

        return loss, cross_entropy, dice
