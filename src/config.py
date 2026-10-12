"""Project paths and default settings."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
DEFAULT_CUSTOMER_COUNT: int = 100_000
DEFAULT_PRODUCT_COUNT: int = 10_000
DEFAULT_ORDER_COUNT: int = 1_000_000
DEFAULT_RANDOM_SEED: int = 42
SPARK_APP_NAME: str = "ecommerce-data-pipeline"


def ensure_data_directories() -> None:
    """Create raw and processed data directories when they do not exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
