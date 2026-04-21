"""
Run once per dataset to generate and persist train/val/test case-ID splits.
Output: data_processed/{dataset}_splits.json
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

# True  = create_test_set(data, seed, ratio=...)
# False = create_test_set(data, ratio=...)  (sepsis hardcodes its own seed)
DATASETS = {
    "bpi12":   ("data.prepare.preprocess_bpi12",   True),
    "bpi17":   ("data.prepare.preprocess_bpi17",   True),
    "sepsis":  ("data.prepare.preprocess_sepsis",  False),
    "traffic": ("data.prepare.preprocess_traffic", True),
}


def get_args():
    parser = argparse.ArgumentParser(description="Generate and save train/val/test splits")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS), help="Dataset name")
    parser.add_argument("--seed",    type=int, default=42, help="Random seed (ignored for sepsis)")
    return parser.parse_args()


def split(data, mod, takes_seed, seed):
    if takes_seed:
        train_ids, test_ids = mod.create_test_set(data, seed)
        train_ids, val_ids  = mod.create_test_set(
            data[data["case:concept:name"].isin(train_ids)], seed, ratio=0.2
        )
    else:
        train_ids, test_ids = mod.create_test_set(data)
        train_ids, val_ids  = mod.create_test_set(
            data[data["case:concept:name"].isin(train_ids)]
        )
    return train_ids, val_ids, test_ids


def main():
    args   = get_args()
    module_name, takes_seed = DATASETS[args.dataset]
    mod    = importlib.import_module(module_name)
    data   = pd.read_csv(
        ROOT / "data_processed" / f"{args.dataset}.csv",
        dtype={"org:resource": str},
    )

    train_ids, val_ids, test_ids = split(data, mod, takes_seed, args.seed)

    out = ROOT / "data_processed" / f"{args.dataset}_splits.json"
    with open(out, "w") as f:
        json.dump(
            {"dataset": args.dataset, "seed": args.seed,
             "train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids},
            f,
        )

    print(f"Saved → {out}")
    print(f"  train: {len(train_ids)}  val: {len(val_ids)}  test: {len(test_ids)}")


if __name__ == "__main__":
    main()
