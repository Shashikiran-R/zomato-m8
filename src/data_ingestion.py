"""
src/data_ingestion.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Final

import pandas as pd

DATASET_NAME: Final[str] = "ManikaSaini/zomato-restaurant-recommendation"

# Mapping from raw HF dataset column names → canonical clean names.
# NOTE: The actual dataset (ManikaSaini/zomato-restaurant-recommendation) uses
#       different column names from what the plan originally assumed.
#       Actual raw schema:
#         'rate'                      -> aggregate_rating  (format: "4.1/5")
#         'approx_cost(for two people)' -> average_cost_for_two
#         'online_order'              -> has_online_delivery
#         'book_table'                -> has_table_booking
COLUMN_RENAME: Final[dict[str, str]] = {
    "name": "name",
    "location": "location",
    "cuisines": "cuisines",
    "approx_cost(for two people)": "average_cost_for_two",
    "rate": "aggregate_rating",
    "votes": "votes",
    "online_order": "has_online_delivery",
    "book_table": "has_table_booking",
}

DEFAULT_OUTPUT: Final[Path] = Path("data/processed/restaurants.parquet")
logger = logging.getLogger(__name__)


def _load_raw_dataset() -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets package.") from exc

    logger.info("Downloading dataset '%s' from Hugging Face ...", DATASET_NAME)
    hf_dataset = load_dataset(DATASET_NAME, trust_remote_code=True)

    if hasattr(hf_dataset, "keys"):
        splits = list(hf_dataset.keys())
        frames = [hf_dataset[split].to_pandas() for split in splits]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = hf_dataset.to_pandas()

    logger.info("Raw dataset loaded: %d records, %d columns.", len(df), len(df.columns))
    return df


def _select_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    available = {col: new for col, new in COLUMN_RENAME.items() if col in df.columns}
    missing = set(COLUMN_RENAME) - set(available)
    if missing:
        logger.warning("Columns not found in dataset: %s", sorted(missing))
    df = df[list(available.keys())].rename(columns=available)
    return df


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    original_count = len(df)

    mandatory = [c for c in ("name", "aggregate_rating", "cuisines") if c in df.columns]
    df = df.dropna(subset=mandatory)
    if "name" in df.columns:
        df = df[df["name"].astype(str).str.strip() != ""]
    if "cuisines" in df.columns:
        df = df[df["cuisines"].astype(str).str.strip().str.lower() != "nan"]
        df = df[df["cuisines"].astype(str).str.strip() != ""]

    if "aggregate_rating" in df.columns:
        # The raw 'rate' field looks like "4.1/5" or "NEW" or "-".
        # Strip everything after (and including) '/' before numeric coercion.
        df["aggregate_rating"] = (
            df["aggregate_rating"]
            .astype(str)
            .str.split("/").str[0]   # keep only the numeric part
            .str.strip()
        )
        df["aggregate_rating"] = pd.to_numeric(df["aggregate_rating"], errors="coerce")
        df = df.dropna(subset=["aggregate_rating"])
        df["aggregate_rating"] = df["aggregate_rating"].clip(lower=0.0, upper=5.0)

    if "average_cost_for_two" in df.columns:
        # Cost field may contain commas as thousands separators (e.g. "1,200").
        df["average_cost_for_two"] = (
            df["average_cost_for_two"]
            .astype(str)
            .str.replace(",", "", regex=False)  # remove thousands separators
            .str.strip()
        )
        df["average_cost_for_two"] = (
            pd.to_numeric(df["average_cost_for_two"], errors="coerce").fillna(0).astype(int)
        )

    if "votes" in df.columns:
        df["votes"] = (
            pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)
        )

    if "cuisines" in df.columns:
        def _norm(raw: object) -> str:
            if pd.isna(raw):
                return ""
            tags = [t.strip().lower() for t in str(raw).split(",")]
            seen: set[str] = set()
            unique: list[str] = []
            for tag in tags:
                if tag and tag not in seen:
                    seen.add(tag)
                    unique.append(tag)
            return ", ".join(unique)
        df["cuisines"] = df["cuisines"].map(_norm)
        df = df[df["cuisines"] != ""]

    for bool_col in ("has_online_delivery", "has_table_booking"):
        if bool_col in df.columns:
            df[bool_col] = (
                df[bool_col].astype(str).str.strip().str.lower().isin({"1", "yes", "true", "y"})
            )

    if "average_cost_for_two" in df.columns:
        def _budget(cost: int) -> str:
            if cost < 500:
                return "low"
            elif cost <= 1500:
                return "medium"
            return "high"
        df["budget"] = df["average_cost_for_two"].map(_budget)
    else:
        df["budget"] = "unknown"

    for str_col in ("name", "location"):
        if str_col in df.columns:
            df[str_col] = df[str_col].astype(str).str.strip()

    if "location" in df.columns:
        df["location"] = df["location"].str.title()

    df = df.reset_index(drop=True)
    logger.info(
        "Cleaning complete: %d records retained out of %d (dropped %d).",
        len(df), original_count, original_count - len(df),
    )
    return df


def _persist(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    size_kb = output_path.stat().st_size / 1024
    logger.info(
        "Parquet written -> '%s'  (%.1f KB, %d records, %d columns).",
        output_path, size_kb, len(df), len(df.columns),
    )


def run_ingestion(
    output_path: Path = DEFAULT_OUTPUT,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Execute the full data-ingestion pipeline."""
    t_start = time.perf_counter()

    if output_path.exists() and not force:
        logger.info("File already exists at '%s'. Use --force to reprocess.", output_path)
        df = pd.read_parquet(output_path, engine="pyarrow")
        logger.info("Loaded existing file: %d records.", len(df))
        return df

    raw_df = _load_raw_dataset()
    selected_df = _select_and_rename(raw_df)
    clean_df = _clean_data(selected_df)
    _persist(clean_df, output_path)

    elapsed = time.perf_counter() - t_start
    logger.info("Pipeline finished in %.2f s. %d restaurants ready.", elapsed, len(clean_df))
    return clean_df


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.data_ingestion",
        description="Download and preprocess the Zomato restaurant dataset into Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, metavar="PATH")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess even if Parquet already exists.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], metavar="LEVEL")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    try:
        df = run_ingestion(output_path=args.output, force=args.force)
        print("\n-- Dataset Summary -------------------------------------------------")
        print(f"  Records       : {len(df):,}")
        print(f"  Columns       : {list(df.columns)}")
        if "aggregate_rating" in df.columns:
            print(f"  Rating range  : {df['aggregate_rating'].min():.1f} - {df['aggregate_rating'].max():.1f}")
        if "budget" in df.columns:
            print(f"  Budget tiers  : {df['budget'].value_counts().to_dict()}")
        if "location" in df.columns:
            print(f"  Unique locs   : {df['location'].nunique()}")
        print(f"  Output file   : {args.output.resolve()}")
        print("--------------------------------------------------------------------\n")
    except Exception:
        logger.exception("Data ingestion failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
