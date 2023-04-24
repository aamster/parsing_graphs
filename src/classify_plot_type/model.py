import lightning
import torch.nn
import torchmetrics


class ClassifyPlotTypeModel(lightning.LightningModule):
    def __init__(
        self,
        learning_rate,
        model: torch.nn.Module
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

    def training_step(self, batch, batch_idx):
        data, target = batch
        logits = self.model(data)
        loss = self._loss_function(logits, target)
        preds = torch.argmax(logits, dim=1)

        self.train_f1.update(preds=preds, target=target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        logits = self.model(data)
        loss = self._loss_function(logits, target)

        preds = torch.argmax(logits, dim=1)
        self.val_precision.update(preds=preds, target=target)
        self.val_recall.update(preds=preds, target=target)
        self.val_f1.update(preds=preds, target=target)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        self.log("val_precision", self.val_precision)
        self.log("val_recall", self.val_recall)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        return optimizer
