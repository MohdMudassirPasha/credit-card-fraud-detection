"""Kaggle dataset download helper.

Automates fetching the real "Credit Card Fraud Detection" dataset
(`mlg-ulb/creditcardfraud`) into ``data/raw/``. Large data files are never
committed to the repository (see ``.gitignore``); this helper reproduces them
on demand.

Requirements
------------
A Kaggle API token must be available at ``~/.kaggle/kaggle.json`` (or via the
``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables). See the README's
"Dataset setup" section for instructions.

Run directly::

    python -m src.data.download
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config, load_config
from src.exceptions import DataDownloadError
from src.logger import get_logger

logger = get_logger(__name__)


def raw_csv_path(config: Config) -> Path:
    """Return the expected path of the raw dataset CSV."""
    return Path(config.data.raw_dir) / config.data.raw_filename


def is_dataset_present(config: Config) -> bool:
    """Return ``True`` if the raw dataset CSV already exists locally."""
    return raw_csv_path(config).is_file()


def download_kaggle_dataset(config: Config | None = None, force: bool = False) -> Path:
    """Download and unzip the Kaggle dataset into ``data/raw/``.

    Parameters
    ----------
    config:
        Loaded configuration. Loaded from disk if omitted.
    force:
        Re-download even if the CSV is already present.

    Returns
    -------
    pathlib.Path
        Path to the downloaded CSV.

    Raises
    ------
    DataDownloadError
        If the Kaggle client/credentials are unavailable or the download fails.
    """
    config = config or load_config()
    target_dir = Path(config.data.raw_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_csv_path(config)

    if csv_path.is_file() and not force:
        logger.info("Dataset already present at %s — skipping download.", csv_path)
        return csv_path

    # Imported lazily so the rest of the project does not hard-depend on the
    # kaggle client (and its credential check at import time).
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataDownloadError(
            "The 'kaggle' package is not installed. Install requirements.txt."
        ) from exc

    logger.info("Authenticating with the Kaggle API...")
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as exc:  # noqa: BLE001 - kaggle raises bare exceptions
        raise DataDownloadError(
            "Kaggle authentication failed. Place a valid kaggle.json at "
            "~/.kaggle/kaggle.json or set KAGGLE_USERNAME / KAGGLE_KEY. "
            "See the README 'Dataset setup' section."
        ) from exc

    logger.info("Downloading '%s' into %s ...", config.data.kaggle_dataset, target_dir)
    try:
        api.dataset_download_files(
            config.data.kaggle_dataset, path=str(target_dir), unzip=True
        )
    except Exception as exc:  # noqa: BLE001 - kaggle raises bare exceptions
        raise DataDownloadError(
            f"Failed to download '{config.data.kaggle_dataset}': {exc}"
        ) from exc

    if not csv_path.is_file():
        raise DataDownloadError(
            f"Download completed but expected file {csv_path} was not found."
        )

    logger.info("Dataset ready at %s", csv_path)
    return csv_path


def main() -> None:
    """CLI entry point: ``python -m src.data.download``."""
    config = load_config()
    download_kaggle_dataset(config)


if __name__ == "__main__":
    main()
