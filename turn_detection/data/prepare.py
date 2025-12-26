import json
import logging
import random
from pathlib import Path
from string import punctuation

import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding

from .download import download_data

logger = logging.getLogger(__name__)


class ChatDataset(Dataset):
    def __init__(self, data_path: str, tokenizer):
        self.samples: list[tuple[str, str, int]] = []
        self.tokenizer = tokenizer
        self._load_data(data_path)

    def _add_samples(self, context: str, full_msg: str):
        # Positive sample
        self.samples.append((context, full_msg, 1))

        # Negative sample (if message has more than one word)
        if len(full_msg.split()) > 1:
            self.samples.append((context, full_msg, 0))

    def _load_data(self, path):
        files = list(Path(path).rglob("*.json"))
        for file in files:
            with open(file, "r") as json_file:
                data = json.load(json_file).get("messages", [])
                for msg1, msg2 in zip(data[:-1], data[1:]):
                    if msg1["role"] == "agent" and msg2["role"] == "client":
                        context = msg1["content"]
                        full_msg = msg2["content"]

                        self._add_samples(context, full_msg)

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
            truncation=False,
            padding=False,
            return_tensors=None,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": label,
        }


class JSONLDataset(Dataset):
    PUNCTUATION = set(punctuation)

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_lines: int | None = None,
    ):
        self.samples: list[tuple[str, str, int]] = []
        self.tokenizer = tokenizer
        self.max_lines = max_lines
        self._load_data(data_path)

    @staticmethod
    def _normalize_turn(turn: str) -> str:
        turn = turn.strip().lstrip("—").replace("-", " ").lower()
        turn = "".join(c for c in turn if c not in JSONLDataset.PUNCTUATION)
        return turn.strip()

    def _load_data(self, path):
        jsonl_path = Path(path) / "conversations.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"File not found: {jsonl_path}. Run download_data first"
            )

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in tqdm(
                enumerate(f), desc="Loading data", total=self.max_lines
            ):
                if self.max_lines is not None and line_idx >= self.max_lines:
                    break

                row = json.loads(line)
                conversation_text = row.get("conversation", "")

                turns = [
                    turn.strip().lstrip("—").strip()
                    for turn in conversation_text.split("\n")
                    if turn.strip()
                ]

                if len(turns) < 2:
                    continue

                for msg1, msg2 in zip(turns[:-1], turns[1:]):
                    full_msg = self._normalize_turn(msg2)
                    if not full_msg:
                        continue

                    context = msg1
                    self.samples.append((context, full_msg, 1))
                    if len(full_msg.split()) > 1:
                        words = full_msg.split()
                        cut_index = random.randint(1, len(words) - 1)
                        neg_msg = " ".join(words[:cut_index])
                        self.samples.append((context, neg_msg, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        context, msg, label = self.samples[index]

        encoding = self.tokenizer(
            context,
            msg,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_tensors=None,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": label,
        }


class InferenceDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer):
        """
        Dataset for inference without labels.

        Args:
            data: List of dicts with 'context' and 'message' keys
            tokenizer: Tokenizer to encode the text
        """
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        encoding = self.tokenizer(
            item["context"],
            item["message"],
            add_special_tokens=True,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }


class EndpointDataModule(pl.LightningDataModule):
    def __init__(self, cfg, predict_data=None):
        super().__init__()
        self.cfg = cfg
        self.train_ds = None
        self.val_ds = None
        self.predict_ds = None
        self.predict_data = predict_data

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
        self.collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt",
        )

        self.type = self.cfg.data.type

    def setup(self, stage=None):
        if stage == "predict":
            if self.predict_data is not None:
                self.predict_ds = InferenceDataset(self.predict_data, self.tokenizer)
            return

        if self.type == "chat":
            full_ds = ChatDataset(
                data_path=Path(self.cfg.data.data_raw_path)
                / self.cfg.data.custom_data_folder,
                tokenizer=self.tokenizer,
            )
        elif self.type == "jsonl":
            download_data(self.cfg)
            full_ds = JSONLDataset(
                data_path=self.cfg.data.data_raw_path,
                tokenizer=self.tokenizer,
                max_lines=self.cfg.data.max_lines,
            )

        val_size = int(self.cfg.data.train_val_split * len(full_ds))
        val_size = min(val_size, self.cfg.data.max_val_lines)
        train_size = len(full_ds) - val_size

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
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.data.eval_batch_size,
            num_workers=self.cfg.data.num_workers,
            collate_fn=self.collator,
        )

    def predict_dataloader(self):
        assert self.predict_ds is not None
        return DataLoader(
            self.predict_ds,
            batch_size=self.cfg.infer.batch_size,
            collate_fn=self.collator,
            num_workers=0,
        )
