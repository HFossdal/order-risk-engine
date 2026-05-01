# Data

Place the IEEE-CIS Fraud Detection CSV files in this directory:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv` *(optional)*
- `test_identity.csv` *(optional)*

## Where to get them

Kaggle competition: <https://www.kaggle.com/competitions/ieee-fraud-detection/data>

If you have the Kaggle CLI configured (`~/.kaggle/kaggle.json`):

```bash
kaggle competitions download -c ieee-fraud-detection -p data/
unzip -o data/ieee-fraud-detection.zip -d data/
```

Otherwise download manually from the Kaggle UI and unzip into this folder.

## Schema (high level)

`train_transaction.csv` — primary table, ~590k rows.
- `TransactionID` (key), `isFraud` (target)
- `TransactionDT` (seconds from a reference timestamp)
- `TransactionAmt`, `ProductCD`
- `card1..card6`, `addr1`, `addr2`
- `P_emaildomain`, `R_emaildomain`
- `C1..C14` (counts), `D1..D15` (timedeltas), `M1..M9` (match flags)
- `V1..V339` (Vesta-engineered features)

`train_identity.csv` — joined on `TransactionID`, ~144k rows.
- `id_01..id_38`, `DeviceType`, `DeviceInfo`

The pipeline left-joins identity onto transaction (most rows have no identity record).
