import json
import logging
import random
from pathlib import Path

import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class ChatDataset(Dataset):
    def __init__(self, data_path: str, tokenizer_name: str, max_length: int = 512):
        self.samples = []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self._load_data(data_path)

    def _load_data(self, path):
        files = list(Path(path).rglob("*.json"))
        for file in files:
            with open(file, "r") as json_file:
                data = json.load(json_file).get("messages", [])
                for i in range(1, len(data)):
                    if data[i]["role"] == "client" and data[i - 1]["role"] == "agent":
                        context = data[i - 1]["content"]
                        full_msg = data[i]["content"]

                        # Positive sample
                        self.samples.append((context, full_msg, 1))

                        # Negative sample (if message has more than one word)
                        if len(full_msg.split()) > 1:
                            self.samples.append((context, full_msg, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        context, msg, label = self.samples[index]

        if label == 0:
            # Randomly cut the message for negative samples
            words = msg.split()
            cut_index = random.randint(1, len(words) - 1)
            msg = " ".join(words[:cut_index])

        # [CLS] context [SEP] msg [SEP]
        encoding = self.tokenizer(
            context,
            msg,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class EndpointDataModule(pl.LightningDataModule):
    def __init__(self, cfg, tokenizer_name: str):
        super().__init__()
        self.cfg = cfg
        self.tokenizer_name = tokenizer_name
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        full_ds = ChatDataset(
            data_path=self.cfg.data.data_path,
            tokenizer_name=self.tokenizer_name,
            max_length=self.cfg.data.max_length,
        )
        train_size = int((1 - self.cfg.data.train_val_split) * len(full_ds))
        val_size = len(full_ds) - train_size
        self.train_ds, self.val_ds = random_split(
            full_ds,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )

        if val_size == 0:
            raise ValueError("Dataset too small or does not exist")

        logger.info(
            f"Dataset split: {train_size} train samples, {val_size} val samples."
        )

    def train_dataloader(self):
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
        )

    def val_dataloader(self):
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
        )

    def test_dataloader(self):
        return self.val_dataloader()
