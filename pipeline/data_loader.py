from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. See data/README.md for download instructions."
        )
    return pd.read_csv(path)


def load_raw(split: str = "train", data_dir: Path | str = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    transactions = _read(data_dir / f"{split}_transaction.csv")
    identity = _read(data_dir / f"{split}_identity.csv")
    return transactions, identity


def merge_transaction_identity(
    transactions: pd.DataFrame, identity: pd.DataFrame
) -> pd.DataFrame:
    return transactions.merge(identity, on="TransactionID", how="left")
