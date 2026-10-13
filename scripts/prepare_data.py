"""
NetGuard AI - Real Dataset Ingestion & Preparation Pipeline
===========================================================
Prepares raw CIC-IDS2017 files (or other network flow CSV captures) for training.
1. Handles multi-file ingestion or single-file processing.
2. Cleans whitespace from column names.
3. Converts string infinities ('Infinity', 'inf') to NaN and validates numerical ranges.
4. Harmonizes label casing and irregular unicode characters (e.g., 'Web Attack – Brute Force').
5. Performs sampling if user specifies a record limit to keep training fast on student hardware.
6. Outputs a cleaned dataset ready for ml.train.

Usage:
    python scripts/prepare_data.py --input path/to/raw_cicids2017.csv --output data/cleaned_dataset.csv --sample 15000
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
from ml.data_loader import load_dataset, validate_schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("scripts.prepare_data")


def prepare_dataset(input_path: str, output_path: str, sample_size: int = None, seed: int = 42):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Loading raw network flows from {input_path}...")
    df = load_dataset(input_path)
    
    schema = validate_schema(df)
    logger.info(f"Raw shape: {schema['num_rows']} rows, {schema['num_columns']} columns.")
    logger.info(f"Detected label column: {schema['label_column']}")

    label_col = schema["label_column"]
    if label_col:
        # Standardize labels
        df[label_col] = df[label_col].astype(str).str.strip()
        # Harmonize common unicode dashes or variants
        df[label_col] = df[label_col].str.replace("–", "-").str.replace("—", "-")
        logger.info(f"Label distribution:\n{df[label_col].value_counts()}")

    # Deduplicate
    initial_count = len(df)
    df = df.drop_duplicates()
    logger.info(f"Deduplication removed {initial_count - len(df)} duplicate flows.")

    # Optional stratified sample if user requested a limit
    if sample_size and len(df) > sample_size:
        logger.info(f"Sampling {sample_size} records from {len(df)} total records...")
        if label_col and df[label_col].nunique() > 1:
            # Stratified sample
            try:
                df = df.groupby(label_col, group_keys=False).apply(
                    lambda x: x.sample(int(np.rint(sample_size * len(x) / len(df))), random_state=seed)
                ).reset_index(drop=True)
            except Exception:
                df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        else:
            df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    # Save prepared dataset
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Cleaned dataset successfully saved to: {output_path} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Ingest and prepare network flow datasets.")
    parser.add_argument("--input", type=str, required=True, help="Path to input raw CSV")
    parser.add_argument("--output", type=str, default="data/processed_dataset.csv", help="Path to output cleaned CSV")
    parser.add_argument("--sample", type=int, default=None, help="Optional record limit for fast training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    prepare_dataset(args.input, args.output, sample_size=args.sample, seed=args.seed)


if __name__ == "__main__":
    main()
