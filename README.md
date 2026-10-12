# E-commerce Data Pipeline

A small Python project that creates realistic fake customers, products, and orders, then uses PySpark to produce business insight tables. Generated source CSV files go in `data/raw/`; Spark output folders go in `data/processed/`.

## Requirements

- Python 3.10 or newer
- Java 11 or newer (required by PySpark)

Install the Python dependencies in a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the pipeline

From the project root, generate the default dataset (100,000 customers, 10,000 products, and 1,000,000 orders):

```bash
python -m src.data_generator
```

Then run the Spark analysis:

```bash
python -m src.spark_analytics
```

You can change the generated row counts and output location:

```bash
python -m src.data_generator --customers 500 --products 50 --orders 2000 --seed 123
python -m src.spark_analytics --input-dir data/raw --output-dir data/processed
```

The analysis writes four CSV output directories: `revenue_by_category`, `monthly_revenue`, `top_products`, and `top_customers`. Cancelled orders are excluded from the revenue analyses.

## Compare Pandas and PySpark performance

To time Parquet loading, joining orders to product prices and calculating revenue, then grouping by customer and printing the top 10:

```bash
python -m src.performance_benchmark --input-dir data/raw
```

The folder must contain `orders.parquet` and `products.parquet`. The benchmark prints elapsed seconds for each stage and the top 10 results from both engines. Spark actions are included in each timed stage so lazy execution is accounted for. Use `--limit-rows 100000` to benchmark a smaller sample.

## Tests

Run the data-generation unit test from the project root:

```bash
python -m unittest discover -s tests
```

## Project layout

```text
genai-pysparl-pipeline/
├── data/
│   ├── raw/                 # Generated source CSV files
│   └── processed/           # Spark insight output folders
├── notebooks/               # Jupyter notebooks
├── src/
│   ├── __init__.py
│   ├── config.py            # Shared paths and defaults
│   ├── data_generator.py    # Synthetic CSV generation
│   └── spark_analytics.py   # PySpark business analysis
├── tests/
│   └── test_data_generator.py
├── .gitignore
├── README.md
└── requirements.txt
```
