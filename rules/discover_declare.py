"""
Discover DECLARE constraints from an event log's training split.

Configuration drives everything — per-template include/min_confidence/category.
The active config is stored inside the output JSON so the checker always
knows what settings produced the rules.

Usage:
    python rules/discover_declare.py --dataset bpi12
    python rules/discover_declare.py --dataset bpi12 --config my_config.json
    python rules/discover_declare.py --dataset bpi12 --save-config   # dump default config and exit
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pm4py

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Default config ─────────────────────────────────────────────────────────────
# Each template entry: include, min_confidence, category
# Discovery mines at min_support + min(min_confidence for included templates),
# then post-filters per template.

DEFAULT_CONFIG = {
    "min_support": 0.1,
    "templates": {
        # -- Immediate ordering (chain constraints) --------------------------------
        "chainresponse":      {"include": True,  "min_confidence": 0.90, "category": "immediate"},
        "chainprecedence":    {"include": True,  "min_confidence": 0.90, "category": "immediate"},
        "chainsuccession":    {"include": True,  "min_confidence": 0.90, "category": "immediate"},
        # -- Eventual positive ordering -------------------------------------------
        "precedence":         {"include": True,  "min_confidence": 0.90, "category": "ordering"},
        "altprecedence":      {"include": True,  "min_confidence": 0.90, "category": "ordering"},
        "altresponse":        {"include": True,  "min_confidence": 0.90, "category": "ordering"},
        "altsuccession":      {"include": True,  "min_confidence": 0.90, "category": "ordering"},
        # -- Cross-path mutual exclusion ------------------------------------------
        # conf=0.80 chosen empirically: gives 22.6% wrong vs 13.5% correct (1.68x)
        "noncoexistence":     {"include": True,  "min_confidence": 0.80, "category": "cross_path"},
        # -- Unary occurrence -----------------------------------------------------
        "init":               {"include": True,  "min_confidence": 0.99, "category": "occurrence"},
        "exactly_one":        {"include": True,  "min_confidence": 0.99, "category": "occurrence"},
        "absence":            {"include": True,  "min_confidence": 0.99, "category": "occurrence"},
        # -- Trace-level (require full trace, skipped in prefix checker) ----------
        "existence":          {"include": False, "min_confidence": 0.90, "category": "trace_level"},
        "responded_existence":{"include": False, "min_confidence": 0.90, "category": "trace_level"},
        "coexistence":        {"include": False, "min_confidence": 0.90, "category": "trace_level"},
        "response":           {"include": False, "min_confidence": 0.90, "category": "trace_level"},
        "succession":         {"include": False, "min_confidence": 0.90, "category": "trace_level"},
        # -- Excluded negatives (anti-discriminative: model already avoids these) -
        "nonchainsuccession": {"include": False, "min_confidence": 0.95, "category": "excluded_negatives"},
        "nonsuccession":      {"include": False, "min_confidence": 0.92, "category": "excluded_negatives"},
    },
    "categories": {
        "immediate":          {"description": "Consecutive-pair ordering — chain constraints only"},
        "ordering":           {"description": "Eventual positive ordering — violations detectable on prefix"},
        "cross_path":         {"description": "Mutual exclusion across process paths — catches cross-branch errors"},
        "occurrence":         {"description": "Unary activity presence/count — fully checkable on prefix"},
        "trace_level":        {"description": "Require full trace — skipped in prefix checker"},
        "excluded_negatives": {"description": "Anti-discriminative: model already avoids these patterns from training"},
    },
}

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


# ── Argument parsing ───────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description="Discover DECLARE rules from a processed event log"
    )
    parser.add_argument("--dataset", type=str, default="bpi12",
                        help="Dataset name (default: bpi12)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a JSON config overriding DEFAULT_CONFIG")
    parser.add_argument("--save-config", action="store_true",
                        help="Write the active config to data/rules/{dataset}_declare_config.json and exit")
    return parser.parse_args()


def load_config(path: str | None) -> dict:
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config
    with open(path) as f:
        override = json.load(f)
    # Deep merge: templates and categories can be partially overridden
    for tmpl, vals in override.get("templates", {}).items():
        config["templates"].setdefault(tmpl, {}).update(vals)
    for cat, vals in override.get("categories", {}).items():
        config["categories"].setdefault(cat, {}).update(vals)
    if "min_support" in override:
        config["min_support"] = override["min_support"]
    return config


# ── Event log loading ──────────────────────────────────────────────────────────

def load_log(dataset: str):
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


# ── Constraint extraction ──────────────────────────────────────────────────────

def to_nl(template_str: str, activities: list[str]) -> str:
    pattern = TEMPLATE_NL.get(template_str.lower(), f"{template_str}({', '.join(activities)})")
    try:
        return pattern.format(*activities)
    except IndexError:
        return f"{template_str}({', '.join(activities)})"


def extract_constraints(declare_model, total_cases: int, config: dict) -> list[dict]:
    """
    Post-filter pm4py results using per-template include/min_confidence from config.
    Only included templates that meet their per-template confidence floor are kept.
    """
    min_support = config["min_support"]
    constraints = []
    i = 0

    for template_str, entries in declare_model.items():
        tmpl_cfg = config["templates"].get(template_str.lower())
        if tmpl_cfg is None or not tmpl_cfg["include"]:
            continue
        min_conf = tmpl_cfg["min_confidence"]
        category = tmpl_cfg["category"]

        for activities, vals in entries.items():
            activities = list(activities) if isinstance(activities, tuple) else [activities]
            support    = round(vals["support"]    / total_cases, 4)
            confidence = round(vals["confidence"] / vals["support"], 4) if vals["support"] else 0.0

            if support < min_support or confidence < min_conf:
                continue

            constraints.append({
                "id":          f"c{i + 1:04d}",
                "template":    template_str,
                "arity":       TEMPLATE_ARITY.get(template_str.lower(), "unknown"),
                "activities":  activities,
                "support":     support,
                "confidence":  confidence,
                "category":    category,
                "description": to_nl(template_str, activities),
            })
            i += 1

    return constraints


# ── Output ─────────────────────────────────────────────────────────────────────

def save_json(constraints: list[dict], dataset: str, config: dict):
    out_dir = ROOT / "data" / "rules"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "dataset":     dataset,
        "config":      config,
        "constraints": constraints,
    }
    path = out_dir / f"{dataset}_declare.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Machine-readable → {path}")


def save_txt(constraints: list[dict], dataset: str, config: dict):
    out_dir = ROOT / "data" / "rules"
    path = out_dir / f"{dataset}_declare.txt"

    by_category: dict[str, list] = {}
    for c in constraints:
        by_category.setdefault(c["category"], []).append(c)

    with open(path, "w") as f:
        f.write(f"DECLARE Rules — {dataset.upper()}\n")
        f.write(f"Min support: {config['min_support']}\n")
        f.write(f"Total constraints: {len(constraints)}\n")
        f.write("=" * 70 + "\n\n")

        for cat in config["categories"]:
            group = sorted(by_category.get(cat, []), key=lambda x: (-x["confidence"], x["template"]))
            if not group:
                continue
            desc = config["categories"][cat]["description"]
            f.write(f"[{cat.upper()}]  —  {len(group)} constraint(s)\n")
            f.write(f"  {desc}\n")
            f.write("-" * 70 + "\n")
            for c in group:
                f.write(f"  {c['description']}\n")
                f.write(f"  support={c['support']:.3f}  confidence={c['confidence']:.3f}\n")
            f.write("\n")

    print(f"  Human-readable   → {path}")


def print_summary(constraints: list[dict], config: dict):
    by_category: dict[str, list] = {}
    for c in constraints:
        by_category.setdefault(c["category"], []).append(c)

    print(f"\n{'=' * 70}")
    print(f"  {len(constraints)} constraints across {len(by_category)} categories")
    print(f"{'=' * 70}")

    for cat in config["categories"]:
        group = by_category.get(cat, [])
        if not group:
            continue
        by_tmpl = Counter(c["template"] for c in group)
        tmpl_summary = ", ".join(f"{t}={n}" for t, n in sorted(by_tmpl.items()))
        print(f"\n[{cat}]  {len(group)} constraints  ({tmpl_summary})")
        print(f"  {config['categories'][cat]['description']}")
        for c in sorted(group, key=lambda x: -x["confidence"])[:3]:
            print(f"  conf={c['confidence']:.3f}  {c['description'][:65]}")
        if len(group) > 3:
            print(f"  ... and {len(group) - 3} more")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    config = load_config(args.config)

    config_out = ROOT / "data" / "rules" / f"{args.dataset}_declare_config.json"
    if args.save_config:
        config_out.parent.mkdir(parents=True, exist_ok=True)
        with open(config_out, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Config written → {config_out}")
        return

    print(f"Loading '{args.dataset}' event log (training split only)...")
    log = load_log(args.dataset)
    total_cases = len(log)
    print(f"  {total_cases} training cases loaded.")

    # Mine at min_support + lowest min_confidence across all templates
    # (pm4py uses a global threshold; per-template filtering happens in extract_constraints)
    all_min_confs = [t["min_confidence"] for t in config["templates"].values()]
    global_min_conf = min(all_min_confs)

    included = [k for k, v in config["templates"].items() if v["include"]]
    excluded = [k for k, v in config["templates"].items() if not v["include"]]
    print(f"\nConfig: min_support={config['min_support']}  global_mining_conf={global_min_conf:.2f}")
    print(f"  Included templates ({len(included)}): {', '.join(sorted(included))}")
    print(f"  Excluded templates ({len(excluded)}): {', '.join(sorted(excluded))}")

    print(f"\nMining DECLARE constraints...")
    declare_model = pm4py.discover_declare(
        log,
        min_support_ratio=config["min_support"],
        min_confidence_ratio=global_min_conf,
    )

    constraints = extract_constraints(declare_model, total_cases, config)

    by_cat = Counter(c["category"] for c in constraints)
    print(f"Found {len(constraints)} constraints after per-template filtering:")
    for cat, n in sorted(by_cat.items()):
        tmpl_min_conf = min(
            v["min_confidence"] for v in config["templates"].values()
            if v["category"] == cat and v["include"]
        )
        print(f"  {cat:20s} {n:4d}  (min_confidence={tmpl_min_conf:.2f})")

    print("\nSaving results...")
    save_json(constraints, args.dataset, config)
    save_txt(constraints, args.dataset, config)
    print_summary(constraints, config)


if __name__ == "__main__":
    main()
