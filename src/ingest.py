from pathlib import Path


def create_dir():
    Path("data/processed/bm25_index").mkdir(parents=True, exist_ok=True)
    Path("data/processed/chunks").mkdir(parents=True, exist_ok=True)