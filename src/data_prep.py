#!/usr/bin/env python3
"""
Data preparation script for Jigsaw Toxic Comments.
Reads data/raw/train.csv (or finds a train.csv under the project),
creates a binary 'target' column, cleans text and writes data/processed/train_clean.parquet
"""

import os
import re
import unicodedata
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# config
DEFAULT_RAW = "data/raw/train.csv"
DEFAULT_OUT = "data/processed/train_clean.parquet"
CHUNK_SIZE = None


# Text Cleaning Func
def clean_text(s: str) -> str:
    """Basic normalization and cleaning while preserving useful punctuation."""
    if not isinstance(s, str):
        return ""
    # unicode normalization
    s = unicodedata.normalize("NFKC", s)

    # replacing URLs, Emails, user mentions
    s = re.sub(r"http\S+|www\.\S+", " <URL> ", s)
    s = re.sub(r"\S+@\S+", " <EMAIL> ", s)    # fixed regex here
    s = re.sub(r"@\w+", " <USER> ", s)

    # replace newlines
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # Remove control characters
    s = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", s)

    # Normalize repeated characters
    s = re.sub(r"(.)\1{3,}", r"\1\1\1", s)

    # Keep most punctuation but remove other special characters
    s = re.sub(r"[^0-9A-Za-z\s\.\,\!\?\:\;\-\(\)\'\"]+", " ", s)

    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()

    # Lowercase
    s = s.lower()
    return s


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    missing = [c for c in label_cols + ['comment_text'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in input: {missing}")

    # creating binary target
    df['target'] = (df[label_cols].sum(axis=1) > 0).astype(int)

    # keep the original text for debugging
    df['comment_raw'] = df['comment_text'].fillna("").astype(str)
    tqdm.pandas(desc="Cleaning text")
    df['comment_clean'] = df['comment_raw'].progress_apply(clean_text)

    # useful metadata
    df['char_len'] = df['comment_clean'].str.len().fillna(0).astype(int)
    df['word_count'] = df['comment_clean'].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)

    # select subset of columns to save
    keep_cols = ['id', 'comment_raw', 'comment_clean', 'char_len', 'word_count', 'target'] + label_cols
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


# main cli
def main(infile: str, outfile: str, chunk_size: int = None, force: bool = False):
    infile_path = Path(infile)
    outfile_path = Path(outfile)
    outfile_path.parent.mkdir(parents=True, exist_ok=True)

    # If infile doesn't exist, try to find any train.csv in project tree
    if not infile_path.exists():
        print(f"[WARN] Provided infile does not exist: {infile_path}")
        cwd = Path.cwd()
        print(f"[INFO] Current working directory: {cwd}")
        # search for train.csv in current tree (project)
        found = list(cwd.rglob("train.csv"))
        if found:
            infile_path = found[0]
            print(f"[INFO] Found train.csv at: {infile_path}  <-- using this file")
        else:
            # if not found, show data/raw contents for debugging and exit
            raw_dir = cwd / "data" / "raw"
            print(f"[ERROR] No train.csv found under current project tree.")
            if raw_dir.exists():
                print(f"[INFO] Files in data/raw:")
                for p in sorted(raw_dir.iterdir()):
                    print("  -", p.name)
            else:
                print(f"[INFO] data/raw directory does not exist here: {raw_dir}")
            raise FileNotFoundError(f"Could not find train.csv. Provide correct --infile or place train.csv in data/raw/")

    if outfile_path.exists() and not force:
        print(f"[INFO] Output already exists: {outfile_path}")
        print("Use --force to overwrite.")
        return

    # Read and process
    if chunk_size:
        print(f"[INFO] Processing in chunks of {chunk_size}")
        chunks = []
        for chunk in pd.read_csv(infile_path, chunksize=chunk_size):
            processed = process_df(chunk)
            chunks.append(processed)
        df_out = pd.concat(chunks, axis=0).reset_index(drop=True)
    else:
        print(f"[INFO] Reading full CSV: {infile_path}")
        df = pd.read_csv(infile_path)
        print(f"[INFO] Raw rows: {len(df)}")
        df_out = process_df(df)

    # Save as parquet for fast downstream loading
    print(f"[INFO] Saving processed data to: {outfile_path} (rows: {len(df_out)})")
    df_out.to_parquet(outfile_path, index=False)
    print("[DONE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Jigsaw Toxic dataset")
    parser.add_argument("--infile", type=str, default=DEFAULT_RAW, help="Path to raw train.csv")
    parser.add_argument("--outfile", type=str, default=DEFAULT_OUT, help="Path to write train_clean.parquet")
    parser.add_argument("--chunk-size", type=int, default=None, help="Optional chunk size for reading CSV")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()
    main(args.infile, args.outfile, chunk_size=args.chunk_size, force=args.force)
