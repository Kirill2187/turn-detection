import logging

import hydra
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from .data import EndpointDataModule
from .models import EndpointClassifier


class CustomProgressBar(RichProgressBar):
    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    logging.basicConfig(level=logging.INFO)

    dm = EndpointDataModule(cfg, tokenizer_name=cfg.model.model_name)
    model = EndpointClassifier(cfg)

    if cfg.task == "train":
        train(cfg, dm, model)


def train(cfg, dm, model):
    logger = MLFlowLogger(
        experiment_name=cfg.train.experiment_name, tracking_uri=cfg.train.mlflow_uri
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.train.ckpt_path, monitor="val_loss", mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        log_every_n_steps=cfg.train.log_every_n_steps,
        logger=logger,
        callbacks=[checkpoint_callback, CustomProgressBar()],
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
