import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pm4py

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Natural language descriptions per DECLARE template (keyed by pm4py's actual template names)
TEMPLATE_NL = {
    "existence":           "'{0}' must occur at least once",
    "absence":             "'{0}' must not occur",
    "exactly_one":         "'{0}' must occur exactly once",
    "init":                "'{0}' must be the first activity in every trace",
    "responded_existence": "If '{0}' occurs, '{1}' must also occur somewhere in the trace",
    "coexistence":         "'{0}' and '{1}' must both occur, or neither occurs",
    "response":            "Whenever '{0}' occurs, '{1}' must eventually follow",
    "precedence":          "'{1}' can only occur if '{0}' has occurred before it",
    "succession":          "Whenever '{0}' occurs '{1}' must follow, and '{1}' only occurs after '{0}'",
    "altresponse":         "Whenever '{0}' occurs, '{1}' must follow before '{0}' can recur",
    "altprecedence":       "Each '{1}' must be preceded by '{0}' with no other '{1}' in between",
    "altsuccession":       "'{0}' and '{1}' alternate: each '{0}' triggers exactly one '{1}' and vice versa",
    "chainresponse":       "Whenever '{0}' occurs, '{1}' must immediately follow",
    "chainprecedence":     "'{1}' can only occur if '{0}' immediately preceded it",
    "chainsuccession":     "'{0}' and '{1}' must always occur as consecutive pairs",
    "noncoexistence":      "'{0}' and '{1}' cannot both occur in the same trace",
    "nonsuccession":       "If '{0}' occurs, '{1}' cannot follow",
    "nonchainsuccession":  "'{0}' and '{1}' cannot occur consecutively",
}

TEMPLATE_ARITY = {
    "existence": "unary", "absence": "unary", "exactly_one": "unary", "init": "unary",
    "responded_existence": "binary", "coexistence": "binary",
    "response": "binary", "precedence": "binary", "succession": "binary",
    "altresponse": "binary", "altprecedence": "binary", "altsuccession": "binary",
    "chainresponse": "binary", "chainprecedence": "binary", "chainsuccession": "binary",
    "noncoexistence": "binary", "nonsuccession": "binary", "nonchainsuccession": "binary",
}

NEGATIVE_TEMPLATES = {"absence", "noncoexistence", "nonsuccession", "nonchainsuccession"}


def get_args():
    parser = argparse.ArgumentParser(description="Discover DECLARE rules from a processed event log")
    parser.add_argument("--dataset",          type=str,   default="bpi12", help="Dataset name (default: bpi12)")
    parser.add_argument("--support",          type=float, default=0.5,     help="Min support ratio (default: 0.5)")
    parser.add_argument("--confidence",       type=float, default=0.8,     help="Min confidence ratio (default: 0.8)")
    parser.add_argument("--positive-only",    action="store_true",         help="Exclude negative constraint templates (absence, noncoexistence, nonsuccession, nonchainsuccession)")
    return parser.parse_args()


def load_log(dataset):
    splits_path = ROOT / "data_processed" / f"{dataset}_splits.json"
    with open(splits_path) as f:
        splits = json.load(f)
    train_ids = set(splits["train_ids"])

    df = pd.read_csv(ROOT / "data_processed" / f"{dataset}.csv", dtype={"org:resource": str})
    df = df[df["case:concept:name"].isin(train_ids)]
    df = pm4py.format_dataframe(
        df,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )
    return pm4py.convert_to_event_log(df)


def to_nl(template_str, activities):
    pattern = TEMPLATE_NL.get(template_str.lower(), f"{template_str}({', '.join(activities)})")
    try:
        return pattern.format(*activities)
    except IndexError:
        return f"{template_str}({', '.join(activities)})"


def extract_constraints(declare_model, total_cases, positive_only=False):
    # pm4py returns {template_str: {(act, ...): {support: count, confidence: count}}}
    constraints = []
    i = 0
    for template_str, entries in declare_model.items():
        if positive_only and template_str.lower() in NEGATIVE_TEMPLATES:
            continue
        for activities, vals in entries.items():
            activities = list(activities) if isinstance(activities, tuple) else [activities]
            support    = round(vals["support"]    / total_cases, 4)
            confidence = round(vals["confidence"] / vals["support"], 4) if vals["support"] else 0.0
            constraints.append({
                "id":          f"c{i+1:04d}",
                "template":    template_str,
                "arity":       TEMPLATE_ARITY.get(template_str.lower(), "unknown"),
                "activities":  activities,
                "support":     support,
                "confidence":  confidence,
                "description": to_nl(template_str, activities),
            })
            i += 1
    return constraints


def save_json(constraints, dataset, support, confidence):
    out_dir = ROOT / "data" / "rules"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "dataset": dataset,
        "parameters": {"min_support": support, "min_confidence": confidence},
        "constraints": constraints,
    }
    path = out_dir / f"{dataset}_declare.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Machine-readable → {path}")


def save_txt(constraints, dataset, support, confidence):
    out_dir = ROOT / "data" / "rules"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset}_declare.txt"

    by_template = {}
    for c in constraints:
        by_template.setdefault(c["template"], []).append(c)

    with open(path, "w") as f:
        f.write(f"DECLARE Rules — {dataset.upper()}\n")
        f.write(f"Min support: {support}  |  Min confidence: {confidence}\n")
        f.write(f"Total constraints discovered: {len(constraints)}\n")
        f.write("=" * 70 + "\n\n")

        for template in sorted(by_template):
            group = sorted(by_template[template], key=lambda x: -x["support"])
            f.write(f"[{template.upper()}]  —  {len(group)} constraint(s)\n")
            f.write("-" * 70 + "\n")
            for c in group:
                f.write(f"  {c['description']}\n")
                f.write(f"  support={c['support']:.2f}  confidence={c['confidence']:.2f}\n")
            f.write("\n")

    print(f"  Human-readable   → {path}")


def print_summary(constraints):
    by_template = {}
    for c in constraints:
        by_template.setdefault(c["template"], []).append(c)

    print(f"\n{'=' * 70}")
    print(f"  {len(constraints)} constraints across {len(by_template)} template(s)")
    print(f"{'=' * 70}")
    for template in sorted(by_template):
        group = sorted(by_template[template], key=lambda x: -x["support"])
        print(f"\n[{template.upper()}]  —  {len(group)} constraint(s)")
        print("-" * 70)
        for c in group[:5]:
            print(f"  {c['description']}")
            print(f"  support={c['support']:.2f}  confidence={c['confidence']:.2f}")
        if len(group) > 5:
            print(f"  ... and {len(group) - 5} more (see .txt file)")


def main():
    args = get_args()
    print(f"Loading '{args.dataset}' event log...")
    log = load_log(args.dataset)
    total_cases = len(log)
    print(f"  {total_cases} cases loaded.")
    print(f"Discovering DECLARE constraints (support≥{args.support}, confidence≥{args.confidence})...")
    declare_model = pm4py.discover_declare(
        log,
        min_support_ratio=args.support,
        min_confidence_ratio=args.confidence,
    )
    constraints = extract_constraints(declare_model, total_cases, positive_only=args.positive_only)
    constraints = [c for c in constraints if c["support"] >= args.support and c["confidence"] >= args.confidence]
    print(f"Found {len(constraints)} constraints.\n")
    print("Saving results...")
    save_json(constraints, args.dataset, args.support, args.confidence)
    save_txt(constraints, args.dataset, args.support, args.confidence)
    print_summary(constraints)


if __name__ == "__main__":
    main()
