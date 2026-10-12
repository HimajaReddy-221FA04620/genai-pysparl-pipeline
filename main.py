"""Generate synthetic e-commerce data and save it as Parquet files."""

import logging
import sys
import time
from pathlib import Path

from src import config
from src.data_generator import SyntheticDataGenerator

LOGGER = logging.getLogger(__name__)


def format_file_size(size_bytes: int) -> str:
    """Format a byte count using a readable binary unit."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def main() -> int:
    """Generate all configured datasets and write each to a Parquet file.

    Returns:
        Process exit code: zero on success and one if generation or writing fails.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    started_at = time.perf_counter()

    try:
        config.ensure_data_directories()
        generator = SyntheticDataGenerator(
            customer_count=config.DEFAULT_CUSTOMER_COUNT,
            product_count=config.DEFAULT_PRODUCT_COUNT,
            order_count=config.DEFAULT_ORDER_COUNT,
            seed=config.DEFAULT_RANDOM_SEED,
        )
        datasets = generator.generate_all()

        output_paths: dict[str, Path] = {}
        for dataset_name, dataframe in datasets.items():
            output_path = config.RAW_DATA_DIR / f"{dataset_name}.parquet"
            dataframe.to_parquet(output_path, index=False)
            output_paths[dataset_name] = output_path

        elapsed = time.perf_counter() - started_at
        print(f"Data generation and Parquet export completed in {elapsed:.2f} seconds.")
        for dataset_name, output_path in output_paths.items():
            print(f"{dataset_name}: {format_file_size(output_path.stat().st_size)} ({output_path})")
        return 0
    except (OSError, ValueError, ImportError) as error:
        LOGGER.exception("Could not generate or save synthetic data: %s", error)
    except Exception as error:
        LOGGER.exception("Unexpected failure while generating synthetic data: %s", error)
    return 1


if __name__ == "__main__":
    sys.exit(main())
