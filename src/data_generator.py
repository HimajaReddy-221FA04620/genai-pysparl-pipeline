"""Generate synthetic e-commerce datasets with Faker, NumPy, and pandas."""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

from src.config import (
    DEFAULT_CUSTOMER_COUNT,
    DEFAULT_ORDER_COUNT,
    DEFAULT_PRODUCT_COUNT,
    DEFAULT_RANDOM_SEED,
    RAW_DATA_DIR,
    ensure_data_directories,
)

LOGGER = logging.getLogger(__name__)
CATEGORIES: tuple[str, ...] = ("Electronics", "Clothing", "Home", "Sports", "Books")
ORDER_STATUSES: tuple[str, ...] = ("placed", "shipped", "delivered", "cancelled")


class SyntheticDataGenerator:
    """Create related, reproducible fake e-commerce data as pandas DataFrames.

    The Pareto-style order assignment reserves 80% of order rows for the most
    popular 20% of customers. Popularity within each group follows a Pareto
    distribution, producing a long-tail customer order pattern.
    """

    def __init__(
        self,
        customer_count: int = 100_000,
        product_count: int = 10_000,
        order_count: int = 1_000_000,
        seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        """Initialize dataset sizes and independent random generators.

        Args:
            customer_count: Number of customers to generate.
            product_count: Number of products to generate.
            order_count: Number of orders to generate.
            seed: Seed for repeatable NumPy and Faker output.
        """
        if min(customer_count, product_count, order_count) < 1:
            raise ValueError("Customer, product, and order counts must all be positive")
        self.customer_count = customer_count
        self.product_count = product_count
        self.order_count = order_count
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.fake = Faker()
        self.fake.seed_instance(seed)

    def generate_customers(self) -> pd.DataFrame:
        """Generate customer identities, demographics, and registration dates.

        Customer ages follow a normal distribution centered at 35 years with
        standard deviation 12, clipped to the range 18 through 90.

        Returns:
            A DataFrame with customer_id, name, email, age, city, country, and
            registration_date columns.
        """
        LOGGER.info("Generating %d customers", self.customer_count)
        names: list[str] = []
        emails: list[str] = []
        cities: list[str] = []
        countries: list[str] = []
        for _ in tqdm(range(self.customer_count), desc="Generating customers", unit="customer"):
            names.append(self.fake.name())
            emails.append(self.fake.unique.email())
            cities.append(self.fake.city())
            countries.append(self.fake.country())

        ages = np.clip(np.rint(self.rng.normal(35, 12, self.customer_count)), 18, 90).astype(np.int16)
        registration_offsets = self.rng.integers(0, 365 * 5, self.customer_count)
        registration_dates = (
            np.datetime64("today", "D") - registration_offsets.astype("timedelta64[D]")
        )
        result = pd.DataFrame({
            "customer_id": np.arange(1, self.customer_count + 1, dtype=np.int64),
            "name": names,
            "email": emails,
            "age": ages,
            "city": cities,
            "country": countries,
            "registration_date": pd.to_datetime(registration_dates),
        })
        LOGGER.info("Generated customer DataFrame with %d rows", len(result))
        return result

    def generate_products(self) -> pd.DataFrame:
        """Generate product names, categories, prices, stock, and ratings.

        Returns:
            A DataFrame with product_id, name, category, price, stock, and
            rating columns. Prices are between 10 and 500 inclusive.
        """
        LOGGER.info("Generating %d products", self.product_count)
        names = [
            f"{self.fake.word().title()} {self.fake.word().title()}"
            for _ in tqdm(range(self.product_count), desc="Generating products", unit="product")
        ]
        result = pd.DataFrame({
            "product_id": np.arange(1, self.product_count + 1, dtype=np.int64),
            "name": names,
            "category": self.rng.choice(CATEGORIES, self.product_count),
            "price": np.round(self.rng.uniform(10, 500, self.product_count), 2),
            "stock": self.rng.integers(0, 1_001, self.product_count),
            "rating": np.round(self.rng.uniform(1, 5, self.product_count), 1),
        })
        LOGGER.info("Generated product DataFrame with %d rows", len(result))
        return result

    def generate_orders(self, products: pd.DataFrame | None = None) -> pd.DataFrame:
        """Generate orders with product references and a Pareto customer split.

        Exactly 80% of order rows (rounded down) are assigned to the top 20%
        of customers. Within the popular and long-tail groups, customers are
        selected with Pareto-distributed weights.

        Args:
            products: Optional product DataFrame from :meth:`generate_products`.
                If omitted, product identifiers and prices are generated for
                order calculations without creating a returned product table.

        Returns:
            A DataFrame containing order_id, customer_id, product_id,
            quantity, order_date, unit_price, total_amount, and status.
        """
        if products is None:
            product_ids = np.arange(1, self.product_count + 1, dtype=np.int64)
            product_prices = self.rng.uniform(10, 500, self.product_count).round(2)
        else:
            if products.empty or not {"product_id", "price"}.issubset(products.columns):
                raise ValueError("products must contain product_id and price columns")
            product_ids = products["product_id"].to_numpy()
            product_prices = products["price"].to_numpy(dtype=float)

        popular_count = max(1, int(np.ceil(self.customer_count * 0.20)))
        popular_customers = np.arange(1, popular_count + 1, dtype=np.int64)
        long_tail_customers = np.arange(popular_count + 1, self.customer_count + 1, dtype=np.int64)
        popular_order_count = int(self.order_count * 0.80)

        LOGGER.info("Generating %d orders with Pareto customer allocation", self.order_count)
        customer_ids = np.empty(self.order_count, dtype=np.int64)
        order_chunk_size = 100_000
        for start in tqdm(range(0, self.order_count, order_chunk_size), desc="Generating orders", unit="order batch"):
            end = min(start + order_chunk_size, self.order_count)
            positions = np.arange(start, end)
            popular_mask = positions < popular_order_count
            customer_ids[start:end][popular_mask] = self._sample_customers(
                popular_customers, int(popular_mask.sum())
            )
            tail_count = int((~popular_mask).sum())
            if tail_count:
                # With fewer than five customers, the popular group can include
                # everyone, so there is no separate long-tail pool.
                pool = long_tail_customers if long_tail_customers.size else popular_customers
                customer_ids[start:end][~popular_mask] = self._sample_customers(pool, tail_count)

        chosen_product_indices = self.rng.integers(0, len(product_ids), self.order_count)
        chosen_product_ids = product_ids[chosen_product_indices]
        unit_prices = product_prices[chosen_product_indices]
        quantities = self.rng.integers(1, 11, self.order_count, dtype=np.int16)
        date_offsets = self.rng.integers(0, 365 * 3, self.order_count)
        order_dates = pd.to_datetime(
            np.datetime64("today", "D") - date_offsets.astype("timedelta64[D]")
        )
        result = pd.DataFrame({
            "order_id": np.arange(1, self.order_count + 1, dtype=np.int64),
            "customer_id": customer_ids,
            "product_id": chosen_product_ids,
            "quantity": quantities,
            "order_date": order_dates,
            # These fields support the companion Spark revenue analysis.
            "unit_price": unit_prices,
            "total_amount": np.round(unit_prices * quantities, 2),
            "status": self.rng.choice(ORDER_STATUSES, self.order_count),
        })
        LOGGER.info("Generated order DataFrame with %d rows", len(result))
        return result

    def generate_all(self) -> dict[str, pd.DataFrame]:
        """Generate all three related DataFrames.

        Returns:
            A dictionary with ``customers``, ``products``, and ``orders`` keys.
        """
        customers = self.generate_customers()
        products = self.generate_products()
        orders = self.generate_orders(products)
        return {"customers": customers, "products": products, "orders": orders}

    def _sample_customers(self, customer_ids: np.ndarray, size: int) -> np.ndarray:
        """Sample customer IDs using normalized Pareto-distributed weights."""
        if size == 0:
            return np.empty(0, dtype=np.int64)
        weights = self.rng.pareto(a=1.16, size=len(customer_ids)) + 1.0
        probabilities = weights / weights.sum()
        return self.rng.choice(customer_ids, size=size, replace=True, p=probabilities)


def generate_data(
    customer_count: int = DEFAULT_CUSTOMER_COUNT,
    product_count: int = DEFAULT_PRODUCT_COUNT,
    order_count: int = DEFAULT_ORDER_COUNT,
    output_dir: Path = RAW_DATA_DIR,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Path]:
    """Generate all datasets and write them as CSV for the Spark pipeline.

    Args:
        customer_count: Number of customer rows.
        product_count: Number of product rows.
        order_count: Number of order rows.
        output_dir: Directory in which the CSV files are written.
        seed: Random seed passed to the generator.

    Returns:
        Paths to the generated customers, products, and orders CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = SyntheticDataGenerator(customer_count, product_count, order_count, seed).generate_all()
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
        LOGGER.info("Wrote %d rows to %s", len(frame), path)
    return paths


def main() -> None:
    """Parse command-line options and generate the requested sample data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=DEFAULT_CUSTOMER_COUNT)
    parser.add_argument("--products", type=int, default=DEFAULT_PRODUCT_COUNT)
    parser.add_argument("--orders", type=int, default=DEFAULT_ORDER_COUNT)
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ensure_data_directories()
    generate_data(args.customers, args.products, args.orders, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
