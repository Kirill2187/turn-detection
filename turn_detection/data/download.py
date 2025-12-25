import logging
from pathlib import Path

from huggingface_hub import snapshot_download
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def download_data(cfg: DictConfig) -> None:
    output_dir = Path(cfg.data.raw_path) / "hf_subset"

    if output_dir.exists() and any(output_dir.iterdir()):
        logger.info(f"Data already exists at {output_dir}, skipping download.")
        return

    logger.info(f"Downloading public data from {cfg.data.hf_repo_id}...")
    snapshot_download(
        repo_id=cfg.data.hf_repo_id,
        repo_type="dataset",
        local_dir=output_dir,
    )
