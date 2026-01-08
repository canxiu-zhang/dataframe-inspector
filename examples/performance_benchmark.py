"""
Performance benchmark for dataframe-inspector with large DataFrames.

This demonstrates how the inspector performs with DataFrames of varying sizes.
"""

import pandas as pd
import time
import io
import contextlib
from dataframe_inspector import Inspector


def generate_nested_data(n_rows):
    """Generate DataFrame with nested structures."""
    return pd.DataFrame(
        {
            "id": range(n_rows),
            "nested": [
                {
                    "user": {"name": f"User{i}", "id": i, "email": f"user{i}@example.com"},
                    "metadata": {"timestamp": f"2024-01-{(i % 28) + 1:02d}", "version": "1.0"},
                    "tags": ["tag1", "tag2", "tag3"],
                }
                for i in range(n_rows)
            ],
        }
    )


def benchmark_overview(df):
    """Benchmark overview() method."""
    start = time.time()
    inspector = Inspector(df)
    with contextlib.redirect_stdout(io.StringIO()):
        inspector.overview()
    elapsed = time.time() - start
    return elapsed


def benchmark_inspect(df, column="nested"):
    """Benchmark inspect_column() method."""
    start = time.time()
    inspector = Inspector(df)
    with contextlib.redirect_stdout(io.StringIO()):
        inspector.inspect_column(column, sample_size=5)
    elapsed = time.time() - start
    return elapsed


if __name__ == "__main__":
    sizes = [100, 1_000, 10_000, 100_000, 1_000_000]

    print("Performance Benchmark for dataframe-inspector")
    print("=" * 60)

    for size in sizes:
        print(f"\nDataFrame size: {size:,} rows")

        # Generate data
        df = generate_nested_data(size)

        # Benchmark overview
        time_overview = benchmark_overview(df)
        print(f"  overview():       {time_overview:.4f}s")

        # Benchmark inspect_column
        time_inspect = benchmark_inspect(df)
        print(f"  inspect_column(): {time_inspect:.4f}s")

    print("\n" + "=" * 60)
    print("Note: inspect_column() samples first 3 non-null rows for key discovery")
    print("      Performance remains constant regardless of DataFrame size")
