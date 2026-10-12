"""Compare Pandas and PySpark timings for the orders Parquet datasets.

Run from the project root with ``python -m src.performance_benchmark``.
The Spark timings include actions so Spark's lazy execution is measured.
"""

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
from pyspark.sql import SparkSession, functions as F

from src.config import RAW_DATA_DIR, SPARK_APP_NAME


def _timed(operation):
    start = perf_counter()
    result = operation()
    return result, perf_counter() - start


def benchmark(input_dir: Path = RAW_DATA_DIR, limit_rows: int | None = None) -> dict[str, dict[str, float]]:
    """Run the three benchmark stages and return elapsed seconds by engine."""
    orders_path = input_dir / "orders.parquet"
    products_path = input_dir / "products.parquet"
    for path in (orders_path, products_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; create Parquet inputs before running the benchmark.")

    results: dict[str, dict[str, float]] = {"Pandas": {}, "PySpark": {}}
    orders_pd, results["Pandas"]["Load"] = _timed(lambda: pd.read_parquet(orders_path))
    products_pd, product_load_time = _timed(lambda: pd.read_parquet(products_path))
    results["Pandas"]["Load"] += product_load_time
    if limit_rows is not None:
        orders_pd = orders_pd.head(limit_rows)

    spark = (
        SparkSession.builder.appName(SPARK_APP_NAME + "-benchmark")
        .master("local[*]")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    orders_sp = products_sp = sales_sp = None
    try:
        def load_spark():
            loaded_orders = spark.read.parquet(str(orders_path))
            loaded_products = spark.read.parquet(str(products_path))
            if limit_rows is not None:
                loaded_orders = loaded_orders.limit(limit_rows)
            loaded_orders = loaded_orders.cache()
            loaded_products = loaded_products.cache()
            # Force the Parquet scans inside the load measurement.
            loaded_orders.count()
            loaded_products.count()
            return loaded_orders, loaded_products

        (orders_sp, products_sp), results["PySpark"]["Load"] = _timed(load_spark)

        def pandas_join():
            return orders_pd.merge(products_pd[["product_id", "price"]], on="product_id").assign(
                revenue=lambda frame: frame["quantity"] * frame["price"]
            )

        sales_pd, results["Pandas"]["Join + revenue"] = _timed(pandas_join)

        def spark_join():
            frame = orders_sp.join(products_sp.select("product_id", "price"), "product_id").withColumn(
                "revenue", F.col("quantity") * F.col("price")
            ).cache()
            frame.count()  # Materialize the join before stopping its timer.
            return frame

        sales_sp, results["PySpark"]["Join + revenue"] = _timed(spark_join)

        def pandas_top10():
            return (
                sales_pd.groupby("customer_id", as_index=False)["revenue"].sum()
                .nlargest(10, "revenue")
            )

        top_pd, results["Pandas"]["Group + top 10"] = _timed(pandas_top10)

        def spark_top10():
            return (
                sales_sp.groupBy("customer_id").agg(F.sum("revenue").alias("total_revenue"))
                .orderBy(F.desc("total_revenue"))
                .limit(10)
                .collect()
            )

        top_sp, results["PySpark"]["Group + top 10"] = _timed(spark_top10)

        _print_comparison(results)
        print("\nPandas top 10 customers:")
        print(top_pd.to_string(index=False))
        print("\nPySpark top 10 customers:")
        for row in top_sp:
            print(f"{row['customer_id']}  {row['total_revenue']:.2f}")
        return results
    finally:
        for frame in (sales_sp, orders_sp, products_sp):
            if frame is not None:
                frame.unpersist()
        spark.stop()


def _print_comparison(results: dict[str, dict[str, float]]) -> None:
    stages = ("Load", "Join + revenue", "Group + top 10")
    print("\nPandas vs PySpark performance (seconds)")
    print(f"{'Operation':<20} {'Pandas':>12} {'PySpark':>12}")
    print("-" * 46)
    for stage in stages:
        print(f"{stage:<20} {results['Pandas'][stage]:>12.3f} {results['PySpark'][stage]:>12.3f}")
    print(f"{'Total':<20} {sum(results['Pandas'].values()):>12.3f} {sum(results['PySpark'].values()):>12.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--limit-rows", type=int, help="Optionally benchmark fewer order rows")
    args = parser.parse_args()
    if args.limit_rows is not None and args.limit_rows < 1:
        parser.error("--limit-rows must be greater than zero")
    benchmark(args.input_dir, args.limit_rows)


if __name__ == "__main__":
    main()
