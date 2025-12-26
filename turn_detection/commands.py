import json
import logging
import subprocess
from pathlib import Path

import fire
import lightning as pl
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import MLFlowLogger
from transformers import AutoTokenizer

from .data import EndpointDataModule
from .models import EndpointClassifier


class CustomProgressBar(RichProgressBar):
    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


def get_git_commit():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def train(config_name="config", resume=None, experiment=None, **kwargs):
    config_path = str(Path(__file__).parent.parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_path):
        override_list = [f"{k}={v}" for k, v in kwargs.items()]
        if experiment:
            override_list = [f"+experiment={experiment}"] + override_list
        cfg = compose(config_name=config_name, overrides=override_list)

    pl.seed_everything(cfg.seed)

    dm = EndpointDataModule(cfg)
    model = (
        EndpointClassifier(cfg)
        if not resume
        else EndpointClassifier.load_from_checkpoint(
            resume, cfg=cfg, weights_only=False
        )
    )

    logger = MLFlowLogger(
        experiment_name=cfg.train.experiment_name,
        tracking_uri=cfg.train.mlflow_uri,
        run_name=cfg.train.run_name,
    )
    logger.experiment.log_param(logger.run_id, "git_commit", get_git_commit())

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.train.ckpt_path,
        monitor="val_roc_auc",
        mode="max",
        save_top_k=1,
        filename="{step}-{val_roc_auc:.4f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_steps=cfg.train.max_steps,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        log_every_n_steps=cfg.train.log_every_n_steps,
        val_check_interval=cfg.train.val_check_interval,
        logger=logger,
        callbacks=[checkpoint_callback, CustomProgressBar(), lr_monitor],
    )

    trainer.fit(model, dm, ckpt_path=resume)


def export_onnx(checkpoint, output="model.onnx", config_name="config", **kwargs):
    config_path = str(Path(__file__).parent.parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_path):
        override_list = [f"{k}={v}" for k, v in kwargs.items()]
        cfg = compose(config_name=config_name, overrides=override_list)

    model = EndpointClassifier.load_from_checkpoint(
        checkpoint, weights_only=False, cfg=cfg, map_location="cpu"
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    dummy_text = "lorem ipsum dolor sit amet"
    dummy_input = tokenizer(dummy_text, return_tensors="pt")

    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=18,
        external_data=False,
    )

    print(f"Model exported to {output}")


def infer(
    input_path, checkpoint, output="predictions.json", config_name="config", **kwargs
):
    config_path = str(Path(__file__).parent.parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = compose(
            config_name=config_name, overrides=[f"{k}={v}" for k, v in kwargs.items()]
        )

    with open(input_path) as f:
        data = json.load(f)

    model = EndpointClassifier.load_from_checkpoint(
        checkpoint, cfg=cfg, map_location="cpu"
    )
    dm = EndpointDataModule(cfg, predict_data=data)
    trainer = pl.Trainer(
        accelerator=cfg.infer.accelerator, devices=cfg.infer.devices, logger=False
    )

    predictions = trainer.predict(model, dm)
    assert predictions is not None

    probs_list = [probs.cpu() for batch in predictions for probs in batch]

    results = [
        {
            **item,
            "probas": {"wait": float(p[0]), "speak": float(p[1])},
        }
        for item, p in zip(data, probs_list)
    ]

    with open(output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Predictions saved to {output}")


def main():
    logging.basicConfig(level=logging.INFO)
    fire.Fire({"train": train, "export_onnx": export_onnx, "infer": infer})


if __name__ == "__main__":
    main()
