from typing import Dict

import lightning
import torch.nn
import torchmetrics


class ClassifyPlotTypeModel(lightning.LightningModule):
    def __init__(
        self,
        learning_rate,
        model: torch.nn.Module,
        hyperparams: Dict
    ):
        super().__init__()
        num_classes = 5
        self._loss_function = torch.nn.CrossEntropyLoss()
        self.model = model

        self.val_precision = torchmetrics.classification.MulticlassPrecision(
            num_classes=num_classes)
        self.val_recall = torchmetrics.classification.MulticlassRecall(
            num_classes=num_classes
        )
        self.train_f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_classes
        )
        self.val_f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_classes
        )
        self._learning_rate = learning_rate
        self._hyperparams = hyperparams

    def training_step(self, batch, batch_idx):
        data, target = batch
        logits = self.model(data)
        loss = self._loss_function(logits, target)
        preds = torch.argmax(logits, dim=1)

        self.train_f1.update(preds, target)
        self.log("train_loss", loss, prog_bar=True)
        self.logger.log_metrics({
            'train_loss': loss,
        }, step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        logits = self.model(data)
        loss = self._loss_function(logits, target)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
        self.val_precision.update(preds=preds, target=target)
        self.val_recall.update(preds=preds, target=target)
        self.val_f1.update(preds=preds, target=target)

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(params=self._hyperparams)

    def on_train_epoch_end(self) -> None:
        self.logger.log_metrics({
            'train_f1': self.train_f1.compute().item()
        }, step=self.current_epoch)
        self.train_f1.reset()

    def on_validation_epoch_end(self) -> None:
        self.logger.log_metrics({
            'val_f1': self.val_f1.compute().item(),
            'val_precision': self.val_precision.compute().item(),
            'val_recall': self.val_recall.compute().item()
        }, step=self.current_epoch)
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        return optimizer
