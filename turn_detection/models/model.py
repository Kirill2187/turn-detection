import lightning as pl
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification


class EndpointClassifier(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.model_name, num_labels=2
        ).train()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(outputs, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(outputs, batch["labels"])

        preds = torch.argmax(outputs, dim=1).detach().cpu()
        labels_cpu = batch["labels"].detach().cpu()

        acc = float(accuracy_score(labels_cpu, preds))
        f1 = float(f1_score(labels_cpu, preds))
        precision = float(precision_score(labels_cpu, preds))
        recall = float(recall_score(labels_cpu, preds))

        try:
            roc_auc = float(
                roc_auc_score(
                    labels_cpu, torch.softmax(outputs, dim=1)[:, 1].detach().cpu()
                )
            )
        except ValueError:
            roc_auc = 0.0

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_roc_auc", roc_auc)
        return loss

    def predict_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"])
        return torch.argmax(outputs, dim=1)

    def configure_optimizers(self):
        return AdamW(
            self.parameters(),
            lr=self.cfg.model.lr,
            weight_decay=self.cfg.model.weight_decay,
        )
