import gzip
import logging
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


def download_data(cfg):
    raw_path = Path(cfg.data.data_path)
    raw_path.mkdir(parents=True, exist_ok=True)

    target_file = raw_path / "conversations.jsonl"

    if target_file.exists():
        logger.info(f"Data already exists at {target_file}, skipping download.")
        return

    gz_path = hf_hub_download(
        repo_id=cfg.data.hf_repo_id,
        filename="conversations.jsonl.gz",
        repo_type="dataset",
        local_dir=raw_path,
    )

    logger.info("Unzipping data...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(target_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    Path(gz_path).unlink()
