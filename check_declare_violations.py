"""
Analyse prefix predictions against DECLARE constraints, broken down by constraint category.

Config is read from the rules JSON produced by discover_declare.py (so checker and
discovery always share the same settings), with optional override via --config.
The active config is embedded in every output file so results are self-documenting.

For every prediction row (correct and wrong) we compute:
  - early_violations:      constraints already violated by the prefix alone
  - prediction_violations: new violations introduced by appending the predicted activity

Usage:
    python check_declare_violations.py
    python check_declare_violations.py --dataset bpi12 --predictions data/BPI12_val_predictions.csv
    python check_declare_violations.py --config my_config.json
    python check_declare_violations.py --save-config
"""

import argparse
import copy
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent

# ── Default config (must match discover_declare.py DEFAULT_CONFIG) ─────────────

DEFAULT_CONFIG = {
    "min_support": 0.1,
    "templates": {
        "chainresponse":      {"include": True,  "min_confidence": 0.90, "category": "immediate"},
        "chainprecedence":    {"include": True,  "min_confidence": 0.90, "category": "immediate"},
        "chainsuccession":    {"include": True,  "min_confidence": 0.90, "category": "immediate"},
        "precedence":         {"include": True,  "min_confidence": 0.90, "category": "ordering"},
        "altprecedence":      {"include": True,  "min_confidence": 0.90, "category": "ordering"},
        "altresponse":        {"include": True,  "min_confidence": 0.90, "category": "ordering"},
        "altsuccession":      {"include": True,  "min_confidence": 0.90, "category": "ordering"},
        "noncoexistence":     {"include": True,  "min_confidence": 0.80, "category": "cross_path"},
        "init":               {"include": True,  "min_confidence": 0.99, "category": "occurrence"},
        "exactly_one":        {"include": True,  "min_confidence": 0.99, "category": "occurrence"},
        "absence":            {"include": True,  "min_confidence": 0.99, "category": "occurrence"},
        "existence":          {"include": False, "min_confidence": 0.90, "category": "trace_level"},
        "responded_existence":{"include": False, "min_confidence": 0.90, "category": "trace_level"},
        "coexistence":        {"include": False, "min_confidence": 0.90, "category": "trace_level"},
        "response":           {"include": False, "min_confidence": 0.90, "category": "trace_level"},
        "succession":         {"include": False, "min_confidence": 0.90, "category": "trace_level"},
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


# ── Argument parsing ───────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description="Check DECLARE violations on prefix predictions, by constraint category"
    )
    parser.add_argument("--dataset", type=str, default="bpi12",
                        help="Dataset name (default: bpi12)")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to predictions CSV. Default: auto-detected test file. "
                             "Pass a validation-set file for methodologically correct tuning.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a JSON config overriding the config embedded in the rules file")
    parser.add_argument("--save-config", action="store_true",
                        help="Write the active config to data/rules/{dataset}_declare_config.json and exit")
    return parser.parse_args()


def load_config(rules_path: Path, override_path: str | None) -> dict:
    """
    Config priority (lowest to highest):
      1. DEFAULT_CONFIG (hardcoded above)
      2. Config stored inside the rules JSON by discover_declare.py
      3. Explicit --config file
    """
    config = copy.deepcopy(DEFAULT_CONFIG)

    if rules_path.exists():
        with open(rules_path) as f:
            rules_data = json.load(f)
        stored = rules_data.get("config", {})
        for tmpl, vals in stored.get("templates", {}).items():
            config["templates"].setdefault(tmpl, {}).update(vals)
        for cat, vals in stored.get("categories", {}).items():
            config["categories"].setdefault(cat, {}).update(vals)
        if "min_support" in stored:
            config["min_support"] = stored["min_support"]

    if override_path:
        with open(override_path) as f:
            override = json.load(f)
        for tmpl, vals in override.get("templates", {}).items():
            config["templates"].setdefault(tmpl, {}).update(vals)
        for cat, vals in override.get("categories", {}).items():
            config["categories"].setdefault(cat, {}).update(vals)
        if "min_support" in override:
            config["min_support"] = override["min_support"]

    return config


# ── Constraint helpers ─────────────────────────────────────────────────────────

def normalize_activity(act: str) -> str:
    return re.sub(r"_(COMPLETE|START|SCHEDULE)$", r"-\1", act.strip())


def parse_prefix(prefix_str: str) -> list[str]:
    return [normalize_activity(a) for a in prefix_str.split(" > ")]


def filter_constraints(constraints: list[dict], config: dict) -> list[dict]:
    """
    Assign category from config, apply per-template confidence floor,
    drop excluded templates. Returns annotated copies.
    """
    result = []
    for c in constraints:
        tmpl = c["template"].lower()
        tmpl_cfg = config["templates"].get(tmpl)
        if tmpl_cfg is None or not tmpl_cfg["include"]:
            continue
        if c["confidence"] < tmpl_cfg["min_confidence"]:
            continue
        c = dict(c)
        c["_category"] = tmpl_cfg["category"]
        result.append(c)
    return result


def category_stats(config: dict) -> tuple[list[str], list[str]]:
    """Return (included_categories, skipped_categories) in config-defined order."""
    included, skipped = [], []
    for cat in config["categories"]:
        if any(v["category"] == cat and v["include"] for v in config["templates"].values()):
            included.append(cat)
        else:
            skipped.append(cat)
    return included, skipped


# ── Constraint checkers ────────────────────────────────────────────────────────

def check_absence(seq, A, **_):
    return A in seq

def check_init(seq, A, **_):
    return len(seq) > 0 and seq[0] != A

def check_noncoexistence(seq, A, B, **_):
    return A in seq and B in seq

def check_nonchainsuccession(seq, A, B, **_):
    return any(seq[i] == A and seq[i + 1] == B for i in range(len(seq) - 1))

def check_nonsuccession(seq, A, B, **_):
    found_A = False
    for act in seq:
        if act == A:
            found_A = True
        elif found_A and act == B:
            return True
    return False

def check_chainprecedence(seq, A, B, **_):
    return any(seq[i] == B and (i == 0 or seq[i - 1] != A) for i in range(len(seq)))

def check_chainresponse(seq, A, B, **_):
    return any(seq[i] == A and seq[i + 1] != B for i in range(len(seq) - 1))

def check_chainsuccession(seq, A, B, **_):
    return check_chainprecedence(seq, A, B) or check_chainresponse(seq, A, B)

def check_exactly_one(seq, A, **_):
    return seq.count(A) > 1

def check_precedence(seq, A, B, **_):
    found_A = False
    for act in seq:
        if act == A:
            found_A = True
        elif act == B and not found_A:
            return True
    return False

def check_altprecedence(seq, A, B, **_):
    last_A = last_B = -1
    for i, act in enumerate(seq):
        if act == A:
            last_A = i
        elif act == B:
            if last_A <= last_B:
                return True
            last_B = i
    return False

def check_altresponse(seq, A, B, **_):
    pending = False
    for act in seq:
        if act == A:
            if pending:
                return True
            pending = True
        elif act == B and pending:
            pending = False
    return False

def check_altsuccession(seq, A, B, **_):
    return check_altprecedence(seq, A, B) or check_altresponse(seq, A, B)


CHECKERS = {
    "absence":            check_absence,
    "init":               check_init,
    "noncoexistence":     check_noncoexistence,
    "nonchainsuccession": check_nonchainsuccession,
    "nonsuccession":      check_nonsuccession,
    "chainprecedence":    check_chainprecedence,
    "chainresponse":      check_chainresponse,
    "chainsuccession":    check_chainsuccession,
    "exactly_one":        check_exactly_one,
    "precedence":         check_precedence,
    "altprecedence":      check_altprecedence,
    "altresponse":        check_altresponse,
    "altsuccession":      check_altsuccession,
}


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_row(prefix_acts: list[str], predicted: str, constraints: list[dict]):
    """
    Returns:
      early_violations:      list of (label, category) already violated by the prefix
      prediction_violations: list of (label, category) newly violated by appending predicted
    """
    extended = prefix_acts + [predicted]
    early, new_viol = [], []

    for c in constraints:
        template = c["template"].lower()
        acts = c["activities"]
        A = acts[0]
        B = acts[1] if len(acts) > 1 else None
        label = f"{c['id']} [{c['template']}]: {c['description']}"
        cat = c["_category"]

        checker = CHECKERS.get(template)
        if checker is None:
            continue

        violated_before = checker(prefix_acts, A=A, B=B) if prefix_acts else False
        violated_after  = checker(extended,     A=A, B=B)

        if violated_before:
            early.append((label, cat))
        elif violated_after:
            new_viol.append((label, cat))

    return early, new_viol


# ── Helpers ────────────────────────────────────────────────────────────────────

def pct(n, base):
    return f"{100 * n / base:.1f}%" if base else "n/a"


def resolve_predictions(args_predictions: str | None, dataset: str) -> Path:
    if args_predictions:
        p = Path(args_predictions)
        if not p.exists():
            raise FileNotFoundError(f"Predictions file not found: {p}")
        return p
    for prefix in (dataset, dataset.upper()):
        p = ROOT / "data" / f"{prefix}_student_model_test_prefix_predictions.csv"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No predictions file found for dataset '{dataset}'. "
        "Pass --predictions <path> explicitly."
    )


def resolve_output(predictions_path: Path, dataset: str) -> Path:
    stem = predictions_path.stem.replace("_prefix_predictions", "")
    return ROOT / "data" / f"{stem}_declare_violation_analysis.csv"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    dataset    = args.dataset
    rules_path = ROOT / "data" / "rules" / f"{dataset}_declare.json"

    config = load_config(rules_path, args.config)

    config_out = ROOT / "data" / "rules" / f"{dataset}_declare_config.json"
    if args.save_config:
        config_out.parent.mkdir(parents=True, exist_ok=True)
        with open(config_out, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Config written → {config_out}")
        return

    with open(rules_path) as f:
        rules_data = json.load(f)

    active_constraints = filter_constraints(rules_data["constraints"], config)
    included_cats, skipped_cats = category_stats(config)
    cat_counts = Counter(c["_category"] for c in active_constraints)

    print(f"Rules file: {rules_path.name}  ({len(rules_data['constraints'])} total constraints)")
    for cat in included_cats:
        min_conf = min(
            v["min_confidence"] for v in config["templates"].values()
            if v["category"] == cat and v["include"]
        )
        print(f"  {cat:20s} {cat_counts[cat]:4d} active  (min_confidence={min_conf:.2f})")
    for cat in skipped_cats:
        n = sum(1 for c in rules_data["constraints"]
                if config["templates"].get(c["template"].lower(), {}).get("category") == cat)
        reason = "included=false"
        print(f"  {cat:20s} {n:4d} skipped ({reason})")
    print(f"  {'Total active':20s} {len(active_constraints):4d}")

    predictions_path = resolve_predictions(args.predictions, dataset)
    out_path = resolve_output(predictions_path, dataset)
    print(f"\nPredictions: {predictions_path.name}")

    with open(predictions_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total        = len(rows)
    wrong_rows   = [r for r in rows if r["predicted_next_activity"] != r["correct_next_activity"]]
    correct_rows = [r for r in rows if r["predicted_next_activity"] == r["correct_next_activity"]]
    print(f"Total predictions: {total}  |  Correct: {len(correct_rows)}  |  Wrong: {len(wrong_rows)}")

    # Accumulators
    stats = {
        split: {"early": {cat: Counter() for cat in included_cats},
                "pred":  {cat: Counter() for cat in included_cats}}
        for split in ("wrong", "correct")
    }
    row_hits = {
        split: {"early": Counter(), "pred": Counter()}
        for split in ("wrong", "correct")
    }

    output_rows = []
    config_json = json.dumps(config, separators=(",", ":"))

    for row in rows:
        prefix_acts = parse_prefix(row["prefix_activities"])
        predicted   = normalize_activity(row["predicted_next_activity"])
        is_correct  = row["predicted_next_activity"] == row["correct_next_activity"]
        split       = "correct" if is_correct else "wrong"

        early, pred_viol = evaluate_row(prefix_acts, predicted, active_constraints)

        for label, cat in early:
            stats[split]["early"][cat][label.split(" ")[0]] += 1
        for label, cat in pred_viol:
            stats[split]["pred"][cat][label.split(" ")[0]] += 1

        for cat in {c for _, c in early}:
            row_hits[split]["early"][cat] += 1
        for cat in {c for _, c in pred_viol}:
            row_hits[split]["pred"][cat] += 1

        def viol_str(pairs, cat):
            items = [label for label, c in pairs if c == cat]
            return " | ".join(items) if items else "none"

        out_row = {
            "prediction_correct":      "yes" if is_correct else "no",
            "prefix_activities":       row["prefix_activities"],
            "predicted_next_activity": row["predicted_next_activity"],
            "correct_next_activity":   row["correct_next_activity"],
            "n_early_violations":      len(early),
            "n_prediction_violations": len(pred_viol),
            "config":                  config_json,
        }
        for cat in included_cats:
            out_row[f"early_{cat}"]      = viol_str(early, cat)
            out_row[f"prediction_{cat}"] = viol_str(pred_viol, cat)

        output_rows.append(out_row)

    fieldnames = [
        "prediction_correct", "prefix_activities", "predicted_next_activity",
        "correct_next_activity", "n_early_violations", "n_prediction_violations", "config",
    ]
    for cat in included_cats:
        fieldnames += [f"early_{cat}", f"prediction_{cat}"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"\nFull analysis saved → {out_path.name}")

    # ── Summary ────────────────────────────────────────────────────────────────
    nw, nc = len(wrong_rows), len(correct_rows)

    print("\n── Statistics by constraint category ───────────────────────────────")
    print(f"{'':27s} {'WRONG':>14}  {'CORRECT':>14}")
    print(f"  predictions           {nw:>8}          {nc:>8}")

    for cat in included_cats:
        desc = config["categories"][cat]["description"]
        min_conf = min(
            v["min_confidence"] for v in config["templates"].values()
            if v["category"] == cat and v["include"]
        )
        n_active = cat_counts[cat]

        ew = row_hits["wrong"]["early"][cat]
        ec = row_hits["correct"]["early"][cat]
        pw = row_hits["wrong"]["pred"][cat]
        pc = row_hits["correct"]["pred"][cat]
        ratio = f"{(pw/nw)/(pc/nc):.2f}x" if pc > 0 and pw > 0 else "—"

        print(f"\n  [{cat}]  n={n_active}  min_conf={min_conf:.2f}")
        print(f"    {desc}")
        print(f"    early violation     {ew:>8} {pct(ew, nw):>6}  {ec:>8} {pct(ec, nc):>6}")
        print(f"    pred. violation     {pw:>8} {pct(pw, nw):>6}  {pc:>8} {pct(pc, nc):>6}  ratio={ratio}")

        top = stats["wrong"]["pred"][cat].most_common(3)
        if top:
            print(f"    top pred. violators (wrong):")
            for cid, count in top:
                c = next((x for x in active_constraints if x["id"] == cid), None)
                if c:
                    print(f"      {cid} {count:>5}x  {c['description'][:70]}")

    if skipped_cats:
        print(f"\n  [skipped — included=false]  " +
              ", ".join(f"{cat} (n={sum(1 for c in rules_data['constraints'] if config['templates'].get(c['template'].lower(), {}).get('category') == cat)})" for cat in skipped_cats))

    # ── Per-constraint discriminability ────────────────────────────────────────
    rows_disc = []
    for cat in included_cats:
        for cid, wrong_abs in stats["wrong"]["pred"][cat].items():
            correct_abs = stats["correct"]["pred"][cat].get(cid, 0)
            net = wrong_abs - correct_abs
            rate_w = wrong_abs / nw if nw else 0
            rate_c = correct_abs / nc if nc else 0
            ratio_val = rate_w / rate_c if rate_c > 0 else float("inf")
            c = next((x for x in active_constraints if x["id"] == cid), None)
            rows_disc.append({
                "id": cid,
                "category": cat,
                "wrong_abs": wrong_abs,
                "correct_abs": correct_abs,
                "net": net,
                "rate_ratio": round(ratio_val, 2),
                "description": c["description"] if c else "",
            })
        # include constraints with zero wrong violations so nothing is silently dropped
        for cid, correct_abs in stats["correct"]["pred"][cat].items():
            if not any(r["id"] == cid for r in rows_disc):
                c = next((x for x in active_constraints if x["id"] == cid), None)
                rows_disc.append({
                    "id": cid,
                    "category": cat,
                    "wrong_abs": 0,
                    "correct_abs": correct_abs,
                    "net": -correct_abs,
                    "rate_ratio": 0.0,
                    "description": c["description"] if c else "",
                })

    rows_disc.sort(key=lambda r: (-r["net"], -r["rate_ratio"]))

    useful = [r for r in rows_disc if r["net"] > 0]
    harmful = [r for r in rows_disc if r["net"] < 0]

    print(f"\n── Per-constraint discriminability (pred violations only) ───────────")
    print(f"  Constraints with any pred violation: {len(rows_disc)}")
    print(f"  Net-positive  (wrong_abs > correct_abs): {len(useful)}")
    print(f"  Net-negative  (wrong_abs < correct_abs): {len(harmful)}")

    if useful:
        print(f"\n  Top net-positive constraints:")
        print(f"  {'id':>6}  {'cat':12}  {'wrong':>5}  {'correct':>7}  {'net':>4}  {'ratio':>6}  description")
        for r in useful[:20]:
            print(f"  {r['id']:>6}  {r['category']:12}  {r['wrong_abs']:>5}  {r['correct_abs']:>7}  "
                  f"{r['net']:>4}  {r['rate_ratio']:>6.2f}x  {r['description'][:55]}")

    disc_path = out_path.with_name(out_path.stem.replace("_declare_violation_analysis", "_declare_discriminability") + ".csv")
    disc_fields = ["id", "category", "wrong_abs", "correct_abs", "net", "rate_ratio", "description"]
    with open(disc_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=disc_fields)
        writer.writeheader()
        writer.writerows(rows_disc)
    print(f"\n  Full per-constraint table → {disc_path.name}")

    # ── Random sample ──────────────────────────────────────────────────────────
    wrong_with_violations = [
        r for r in output_rows
        if r["prediction_correct"] == "no"
        and (r["n_early_violations"] > 0 or r["n_prediction_violations"] > 0)
    ]
    if wrong_with_violations:
        print("\n── Random sample of wrong predictions with violations ───────────────")
        for i, r in enumerate(random.sample(wrong_with_violations, min(2, len(wrong_with_violations))), 1):
            prefix_acts = r["prefix_activities"].split(" > ")
            print(f"\n[{i}] prefix ({len(prefix_acts)} steps): "
                  f"predicted={r['predicted_next_activity']}  correct={r['correct_next_activity']}")
            for cat in included_cats:
                if (pv := r[f"prediction_{cat}"]) != "none":
                    print(f"     pred  [{cat}]: {pv[:120]}")
                if (ev := r[f"early_{cat}"]) != "none":
                    print(f"     early [{cat}]: {ev[:120]}")


if __name__ == "__main__":
    main()
