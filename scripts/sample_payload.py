"""Print sample JSON payloads for POST /predict.

Usage:
    python scripts/sample_payload.py            # benign-looking
    python scripts/sample_payload.py --risky    # something the model should flag

Pipe into curl:
    python scripts/sample_payload.py | curl -X POST http://localhost:8000/predict \
        -H 'Content-Type: application/json' -d @-

Note the C* and D* fields. These are Vesta's anonymized count / timedelta
features and they carry most of the signal in the IEEE-CIS dataset. In
production they'd be served from a feature store keyed on `card1`; here we
include realistic values so the demo actually discriminates between orders.
Without them the model has very little to go on and most orders score near
zero.
"""

import argparse
import json

BENIGN = {
    "TransactionAmt": 89.50,
    "TransactionDT": 14 * 3600,        # mid-afternoon
    "ProductCD": "W",
    "card1": 13926, "card2": 583.0, "card3": 150.0,
    "card4": "visa", "card5": 226.0, "card6": "credit",
    "addr1": 315.0, "addr2": 87.0,
    "P_emaildomain": "gmail.com",
    "DeviceType": "desktop",
    "DeviceInfo": "Windows",
    "extra": {
        "C1": 1, "C2": 1, "C13": 1, "C14": 1,
        "D1": 14, "D2": 14, "D15": 14,
        "M4": "M2", "M6": "T",
        "id_31": "chrome 78.0",
    },
}

RISKY = {
    "TransactionAmt": 1899.99,
    "TransactionDT": 3 * 3600,         # 3am UTC
    "ProductCD": "C",
    "card1": 9500, "card2": 321.0, "card3": 185.0,
    "card4": "visa", "card5": 226.0, "card6": "debit",
    "addr1": 204.0, "addr2": 87.0,
    "P_emaildomain": "anonymous.com",
    "R_emaildomain": "anonymous.com",
    "DeviceType": "mobile",
    "DeviceInfo": "SAMSUNG SM-G930V Build/NRD90M",
    "extra": {
        # high recent counts on this card -> velocity signal
        "C1": 12, "C2": 12, "C13": 18, "C14": 14,
        # very recent first-seen / last-seen
        "D1": 0, "D2": 0, "D15": 0,
        "M4": "M0", "M6": "F",
        "id_31": "samsung browser 4.0",
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--risky", action="store_true", help="emit a higher-risk payload")
    args = parser.parse_args()
    print(json.dumps(RISKY if args.risky else BENIGN, indent=2))
