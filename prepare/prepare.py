"""
Convert a raw XES event log into a processed CSV ready for split generation
and model training.

Usage:
    python prepare/prepare.py --dataset sepsis
    python prepare/prepare.py --dataset bpi12 --input path/to/file.xes.gz

The --input flag overrides the default xes_file path from datasets/config.py.
"""

import argparse
import gzip
import os
import shutil
import tempfile

import pandas as pd
import pm4py

from datasets.config import DATASET_REGISTRY


# ── XES reader ────────────────────────────────────────────────────────────────

def read_xes(path: str) -> pd.DataFrame:
    """Read an XES or XES.gz file and return a flat DataFrame via pm4py."""
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f_in:
            with tempfile.NamedTemporaryFile(suffix=".xes", delete=False) as f_out:
                shutil.copyfileobj(f_in, f_out)
                tmp = f_out.name
        try:
            log = pm4py.read_xes(tmp)
        finally:
            os.unlink(tmp)
    else:
        log = pm4py.read_xes(path)

    df = pm4py.convert_to_dataframe(log)
    return df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)


# ── Dataset-specific transforms ───────────────────────────────────────────────
# Each function receives the raw pm4py DataFrame (already sorted) and must return
# a DataFrame that includes at minimum:
#   concept:name, case:concept:name, time:timestamp, label
# Temporal features (elapsed_time, time_since_previous) are added after the transform.

def _transform_bpi12(df: pd.DataFrame) -> pd.DataFrame:
    # Merge activity + lifecycle into a single token e.g. "A_SUBMITTED-COMPLETE"
    df["concept:name"] = df["concept:name"] + "-" + df["lifecycle:transition"]
    df = df.drop(columns=["lifecycle:transition"])

    # Normalise org:resource to plain integer strings ("112.0" → "112")
    df["org:resource"] = df["org:resource"].apply(
        lambda x: str(int(float(x))) if pd.notna(x) and str(x) not in ("nan", "None") else "UNKNOWN"
    )

    # Label: 1 if A_ACCEPTED-COMPLETE appears anywhere in the case
    accepted = set(df[df["concept:name"] == "A_ACCEPTED-COMPLETE"]["case:concept:name"])
    df["label"] = df["case:concept:name"].isin(accepted).astype(int)
    return df


def _transform_bpi17(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("Add BPI17 XES transform here")


def _transform_sepsis(df: pd.DataFrame) -> pd.DataFrame:
    # pm4py prefixes case attributes with "case:" — rename the ones the
    # preprocessor expects without that prefix.
    case_attr_renames = {
        "case:Age": "Age",
        "case:Diagnose": "Diagnose",
        "case:DiagnosticArtMedic": "DiagnosticArtMedic",
        "case:DiagnosticBlood": "DiagnosticBlood",
        "case:DiagnosticECG": "DiagnosticECG",
        "case:DiagnosticIC": "DiagnosticIC",
        "case:DiagnosticLacticAcid": "DiagnosticLacticAcid",
        "case:DiagnosticLiquor": "DiagnosticLiquor",
        "case:DiagnosticNeutrophils": "DiagnosticNeutrophils",
        "case:DiagnosticSputum": "DiagnosticSputum",
        "case:DiagnosticUrinaryCulture": "DiagnosticUrinaryCulture",
        "case:DiagnosticUrinarySediment": "DiagnosticUrinarySediment",
        "case:DiagnosticXthorax": "DiagnosticXthorax",
        "case:DisfuncOrg": "DisfuncOrg",
        "case:Hypotension": "Hypotension",
        "case:Hypoxie": "Hypoxie",
        "case:InfectionSuspected": "InfectionSuspected",
        "case:Infusion": "Infusion",
        "case:Oligurie": "Oligurie",
        "case:SIRScriteria2": "SIRScriteria2",
        "case:SIRScriteria3": "SIRScriteria3",
        "case:SIRScriteria4": "SIRScriteria4",
    }
    rename_map = {k: v for k, v in case_attr_renames.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Label: 1 = uncomplicated release (Release A or Release B)
    released = set(df[df["concept:name"].isin({"Release A", "Release B"})]["case:concept:name"])
    df["label"] = df["case:concept:name"].isin(released).astype(int)

    # Rule flags (trace-level, stored on every row for stratified splitting)
    # rule_1: IV Antibiotics was administered at some point
    iv_cases = set(df[df["concept:name"] == "IV Antibiotics"]["case:concept:name"])
    df["rule_1"] = df["case:concept:name"].isin(iv_cases).astype(int)

    # rule_2: LacticAcid lab event present
    la_cases = set(df[df["concept:name"] == "LacticAcid"]["case:concept:name"])
    df["rule_2"] = df["case:concept:name"].isin(la_cases).astype(int)

    # rule_3: CRP event occurs before IV Antibiotics (protocol order)
    def _crp_before_atb(grp):
        crp_idx = grp.index[grp["concept:name"] == "CRP"].tolist()
        atb_idx = grp.index[grp["concept:name"] == "IV Antibiotics"].tolist()
        return int(bool(crp_idx and atb_idx and min(crp_idx) < min(atb_idx)))
    rule3 = df.groupby("case:concept:name", sort=False).apply(_crp_before_atb)
    df["rule_3"] = df["case:concept:name"].map(rule3).fillna(0).astype(int)

    return df


def _transform_traffic(df: pd.DataFrame) -> pd.DataFrame:
    # pm4py prefixes case attributes — rename to match what the preprocessor expects
    case_attr_renames = {
        "case:amount": "amount",
        "case:article": "article",
        "case:dismissal": "dismissal",
        "case:expense": "expense",
        "case:lastSent": "lastSent",
        "case:matricola": "matricola",
        "case:notificationType": "notificationType",
        "case:paymentAmount": "paymentAmount",
        "case:points": "points",
        "case:totalPaymentAmount": "totalPaymentAmount",
        "case:vehicleClass": "vehicleClass",
    }
    rename_map = {k: v for k, v in case_attr_renames.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Label: 1 if fine was eventually paid
    paid = set(df[df["concept:name"] == "Payment"]["case:concept:name"])
    df["label"] = df["case:concept:name"].isin(paid).astype(int)

    # rule_1: Add penalty event is present (penalty was triggered)
    penalty_cases = set(df[df["concept:name"] == "Add penalty"]["case:concept:name"])
    df["rule_1"] = df["case:concept:name"].isin(penalty_cases).astype(int)

    # rule_2: Send Fine notification was sent
    sent_cases = set(df[df["concept:name"] == "Send Fine"]["case:concept:name"])
    df["rule_2"] = df["case:concept:name"].isin(sent_cases).astype(int)

    # rule_3: Insert Fine Notification is present
    notif_cases = set(df[df["concept:name"] == "Insert Fine Notification"]["case:concept:name"])
    df["rule_3"] = df["case:concept:name"].isin(notif_cases).astype(int)

    return df


_TRANSFORMS = {
    "bpi12": _transform_bpi12,
    "bpi17": _transform_bpi17,
    "sepsis": _transform_sepsis,
    "traffic": _transform_traffic,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare a raw XES event log into a processed CSV for training"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset name",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to the XES or XES.gz file (overrides the registry default)",
    )
    return parser.parse_args()


def main():
    args = get_args()
    cfg = DATASET_REGISTRY[args.dataset]

    xes_path = args.input or cfg["xes_file"]
    if not os.path.exists(xes_path):
        raise FileNotFoundError(
            f"XES file not found: {xes_path}\n"
            f"Download it and either place it at that path or pass --input <path>."
        )

    print(f"[{args.dataset}] Reading XES: {xes_path}")
    df = read_xes(xes_path)
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    print(f"[{args.dataset}] Applying transform…")
    df = _TRANSFORMS[args.dataset](df)

    # Temporal features (common to all datasets)
    df["elapsed_time"] = df.groupby("case:concept:name")["time:timestamp"].transform(
        lambda x: (x - x.min()).dt.total_seconds() / 60
    )
    df["time_since_previous"] = (
        df.groupby("case:concept:name")["time:timestamp"]
        .diff()
        .dt.total_seconds()
        .fillna(0)
        / 60
    )

    os.makedirs("data_processed", exist_ok=True)
    output_path = f"data_processed/{args.dataset}.csv"
    df.to_csv(output_path, index=False)

    print(f"[{args.dataset}] Saved → {output_path}")
    print(f"  Final shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"\n  Label distribution:\n{df.groupby('case:concept:name')['label'].first().value_counts()}")
    print(f"\n  Sample activities:\n{df['concept:name'].unique()[:10]}")


if __name__ == "__main__":
    main()
