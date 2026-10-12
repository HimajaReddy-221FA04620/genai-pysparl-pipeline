"""PySpark analytics for synthetic e-commerce orders and product data."""

import argparse
import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, SPARK_APP_NAME, ensure_data_directories

LOGGER = logging.getLogger(__name__)


class SalesAnalytics:
    """Load Parquet datasets and calculate sales summaries with PySpark."""

    def __init__(self, spark: SparkSession | None = None) -> None:
        """Initialize analytics, optionally with an existing Spark session."""
        self.spark = spark

    def create_spark_session(self, app_name: str = SPARK_APP_NAME) -> SparkSession:
        """Create a local Spark session configured for 4 GB and Kryo.

        Adaptive query execution is enabled to let Spark optimize query plans
        at runtime. Calling the method again reuses this instance's session.
        """
        self.spark = (
            SparkSession.builder
            .appName(app_name)
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .getOrCreate()
        )
        LOGGER.info("Created local Spark session: %s", app_name)
        return self.spark

    def load_parquet(self, path: str | Path) -> DataFrame:
        """Load a Parquet file or directory into a Spark DataFrame.

        Args:
            path: Local or Spark-compatible path to the Parquet dataset.

        Returns:
            The loaded Spark DataFrame.

        Raises:
            RuntimeError: If this analytics object has no Spark session.
        """
        if self.spark is None:
            raise RuntimeError("Create a Spark session before loading Parquet data")
        LOGGER.info("Loading Parquet dataset from %s", path)
        return self.spark.read.parquet(str(path))

    def top_customers_by_revenue(
        self, orders_df: DataFrame, products_df: DataFrame, n: int = 10
    ) -> DataFrame:
        """Return the top N customers ranked by order revenue.

        Revenue is calculated as order quantity multiplied by product price.
        The orders input must contain ``customer_id``, ``product_id``, and
        ``quantity``; products must contain ``product_id`` and ``price``.
        """
        if n < 1:
            raise ValueError("n must be greater than zero")
        sales = self._join_sales(orders_df, products_df)
        return (
            sales.groupBy("customer_id")
            .agg(F.round(F.sum("revenue"), 2).alias("total_revenue"))
            .orderBy(F.desc("total_revenue"))
            .limit(n)
        )

    def sales_by_category(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Return revenue and units sold grouped by product category."""
        sales = self._join_sales(orders_df, products_df)
        return (
            sales.groupBy("category")
            .agg(
                F.round(F.sum("revenue"), 2).alias("total_revenue"),
                F.sum("quantity").alias("units_sold"),
            )
            .orderBy(F.desc("total_revenue"))
        )

    def monthly_trends(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Return monthly revenue and month-over-month growth percentage.

        The first month has a null growth rate because no prior month exists.
        Growth is also null when prior-month revenue is zero.
        """
        if "order_date" not in orders_df.columns:
            raise ValueError("orders_df must contain an order_date column for monthly trends")
        sales = self._join_sales(orders_df, products_df)
        monthly = (
            sales.withColumn("month", F.date_format(F.to_timestamp("order_date"), "yyyy-MM"))
            .groupBy("month")
            .agg(F.sum("revenue").alias("revenue"))
        )
        month_window = Window.orderBy("month")
        with_previous = monthly.withColumn("previous_revenue", F.lag("revenue").over(month_window))
        return with_previous.select(
            "month",
            F.round("revenue", 2).alias("revenue"),
            F.when(
                F.col("previous_revenue").isNotNull() & (F.col("previous_revenue") != 0),
                F.round(
                    (F.col("revenue") - F.col("previous_revenue"))
                    / F.col("previous_revenue") * F.lit(100),
                    2,
                ),
            ).otherwise(F.lit(None).cast("double")).alias("revenue_growth_pct"),
        ).orderBy("month")

    @staticmethod
    def _join_sales(orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Join order lines to product prices and add a revenue column."""
        required_orders = {"customer_id", "product_id", "quantity"}
        required_products = {"product_id", "price", "category"}
        missing_orders = required_orders.difference(orders_df.columns)
        missing_products = required_products.difference(products_df.columns)
        if missing_orders or missing_products:
            raise ValueError(
                f"Missing columns; orders: {sorted(missing_orders)}, "
                f"products: {sorted(missing_products)}"
            )
        return (
            orders_df.join(products_df.select("product_id", "price", "category"), "product_id")
            .withColumn("revenue", F.col("quantity") * F.col("price"))
        )


def create_spark_session(app_name: str = SPARK_APP_NAME) -> SparkSession:
    """Create or retrieve a configured local Spark session for the pipeline."""
    return SalesAnalytics().create_spark_session(app_name)


def load_datasets(spark: SparkSession, input_dir: Path = RAW_DATA_DIR) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Load customer, product, and order datasets from Parquet or CSV files.

    Parquet is preferred when the matching ``<dataset>.parquet`` path exists;
    CSV remains supported for data produced by the earlier generator workflow.
    """
    frames = []
    for dataset in ("customers", "products", "orders"):
        parquet_path = input_dir / f"{dataset}.parquet"
        csv_path = input_dir / f"{dataset}.csv"
        if parquet_path.exists():
            frame = spark.read.parquet(str(parquet_path))
            path = parquet_path
        elif csv_path.is_file():
            frame = spark.read.option("header", True).option("inferSchema", True).csv(str(csv_path))
            path = csv_path
        else:
            raise FileNotFoundError(
                f"Missing {dataset} dataset at {parquet_path} or {csv_path}. Generate data first."
            )
        frames.append(frame)
        LOGGER.info("Loaded %s from %s", dataset, path)
    return frames[0], frames[1], frames[2]


def build_insights(customers: DataFrame, products: DataFrame, orders: DataFrame) -> dict[str, DataFrame]:
    """Build business insight tables from the three input DataFrames."""
    valid_orders = orders.filter(F.col("status") != "cancelled")
    enriched = valid_orders.join(products, "product_id").join(customers, "customer_id")
    return {
        "revenue_by_category": enriched.groupBy("category").agg(
            F.round(F.sum("total_amount"), 2).alias("revenue"),
            F.sum("quantity").alias("units_sold"), F.countDistinct("order_id").alias("order_count"),
        ).orderBy(F.desc("revenue")),
        "monthly_revenue": valid_orders.withColumn("month", F.date_format(F.to_timestamp("order_date"), "yyyy-MM"))
            .groupBy("month").agg(F.round(F.sum("total_amount"), 2).alias("revenue"))
            .orderBy("month"),
        "top_products": enriched.groupBy("product_id", "name", "category").agg(
            F.sum("quantity").alias("units_sold"), F.round(F.sum("total_amount"), 2).alias("revenue"),
        ).orderBy(F.desc("revenue")).limit(10),
        "top_customers": enriched.groupBy("customer_id", "name", "email").agg(
            F.round(F.sum("total_amount"), 2).alias("lifetime_value"), F.countDistinct("order_id").alias("order_count"),
        ).orderBy(F.desc("lifetime_value")).limit(10),
    }


def save_insights(insights: dict[str, DataFrame], output_dir: Path = PROCESSED_DATA_DIR) -> None:
    """Save each insight DataFrame as an overwrite-mode CSV directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in insights.items():
        destination = output_dir / name
        frame.coalesce(1).write.mode("overwrite").option("header", True).csv(str(destination))
        LOGGER.info("Saved %s to %s", name, destination)


def analyze(input_dir: Path = RAW_DATA_DIR, output_dir: Path = PROCESSED_DATA_DIR) -> dict[str, DataFrame]:
    """Load source data, calculate insight tables, save them, and return them."""
    spark = create_spark_session()
    try:
        customers, products, orders = load_datasets(spark, input_dir)
        insights = build_insights(customers, products, orders)
        save_insights(insights, output_dir)
        return insights
    finally:
        spark.stop()


def main() -> None:
    """Parse command-line paths and run the Spark analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ensure_data_directories()
    analyze(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
