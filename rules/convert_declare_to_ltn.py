"""
Convert DECLARE constraints to an importable LTN constraints module.

Output: data/rules/{dataset}_ltn_constraints.py

Two selection modes
-------------------
Template mode (default):
    Filters by template name and confidence — existence-based templates only.

Discriminability mode (--discriminability-csv):
    Reads the per-constraint discriminability CSV produced by check_declare_violations.py
    and keeps every constraint with net >= --min-net (i.e. more wrong predictions
    violated it than correct ones).  All templates are supported.

The generated module supports all three LTN integration levels:
  Level 1 (input):        compute_level1_features(x, activity_vocab, activity_col_start, seq_len)
  Level 2 (loss):         build_level2_formulas(...)   [existence-based templates only]
  Level 3 (architecture): make_predicates(...)  /  make_constants(activity_vocab)

Usage:
    python rules/convert_declare_to_ltn.py --dataset bpi12
    python rules/convert_declare_to_ltn.py --dataset bpi12 --templates responded_existence coexistence
    python rules/convert_declare_to_ltn.py --dataset bpi12 --min-confidence 0.8 --top-k 20
    python rules/convert_declare_to_ltn.py --dataset bpi12 \\
        --discriminability-csv data/BPI12_student_model_val_declare_discriminability.csv \\
        --min-net 1
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent

EXISTENCE_TEMPLATES = {
    "existence",
    "exactly_one",
    "init",
    "responded_existence",
    "coexistence",
}

ALL_SUPPORTED_TEMPLATES = EXISTENCE_TEMPLATES | {
    "absence",
    "noncoexistence",
    "precedence",
    "altresponse",
    "altprecedence",
    "altsuccession",
    "chainresponse",
    "chainprecedence",
    "chainsuccession",
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert DECLARE rules to LTN constraints module"
    )
    parser.add_argument(
        "--dataset", type=str, default="bpi12",
        help="Dataset name (default: bpi12)",
    )
    # ── Template mode ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--templates", type=str, nargs="+", default=["all"],
        help=(
            "Constraint templates to include (template mode only). "
            "Use 'all' for all existence-based templates, or list specific names. "
            f"Default: all"
        ),
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.0,
        help="Only include constraints with confidence >= this value (default: 0.0)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Keep at most top-K constraints per template, ranked by confidence (default: no limit)",
    )
    # ── Discriminability mode ──────────────────────────────────────────────────
    parser.add_argument(
        "--discriminability-csv", type=str, default=None,
        help=(
            "Path to per-constraint discriminability CSV from check_declare_violations.py. "
            "When supplied, template/confidence/top-k filters are ignored and every "
            "constraint with net >= --min-net is included."
        ),
    )
    parser.add_argument(
        "--min-net", type=int, default=1,
        help="Minimum absolute net (wrong_abs - correct_abs) to include (default: 1)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output file path (default: data/rules/{dataset}_ltn_constraints.py)",
    )
    return parser.parse_args()


# ── Constraint loading ─────────────────────────────────────────────────────────

def load_by_discriminability(dataset: str, disc_csv: str, min_net: int) -> list[dict]:
    disc_path = Path(disc_csv)
    if not disc_path.exists():
        raise FileNotFoundError(f"Discriminability CSV not found: {disc_path}")

    with open(disc_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    kept_ids = {
        r["id"] for r in rows
        if int(r["net"]) >= min_net
        and r["id"]  # skip blank ids
    }

    rules_path = ROOT / "data" / "rules" / f"{dataset}_declare.json"
    if not rules_path.exists():
        raise FileNotFoundError(f"Rules file not found: {rules_path}")
    with open(rules_path) as f:
        data = json.load(f)

    filtered = [c for c in data["constraints"] if c["id"] in kept_ids]
    unsupported = {c["template"] for c in filtered} - ALL_SUPPORTED_TEMPLATES
    if unsupported:
        print(f"  Warning: dropping {sum(1 for c in filtered if c['template'] in unsupported)} "
              f"constraints with unsupported templates: {unsupported}")
        filtered = [c for c in filtered if c["template"] in ALL_SUPPORTED_TEMPLATES]

    return filtered


def load_by_template(dataset: str, templates: set[str], min_confidence: float,
                     top_k: int | None) -> list[dict]:
    path = ROOT / "data" / "rules" / f"{dataset}_declare.json"
    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {path}")
    with open(path) as f:
        data = json.load(f)

    filtered = [
        c for c in data["constraints"]
        if c["template"] in templates and c["confidence"] >= min_confidence
    ]

    if top_k is not None:
        by_template: dict[str, list] = {}
        for c in filtered:
            by_template.setdefault(c["template"], []).append(c)
        filtered = []
        for template_constraints in by_template.values():
            ranked = sorted(template_constraints, key=lambda c: -c["confidence"])
            filtered.extend(ranked[:top_k])

    return filtered


def resolve_templates(templates_arg: list[str]) -> set[str]:
    if templates_arg == ["all"]:
        return EXISTENCE_TEMPLATES
    requested = {t.lower() for t in templates_arg}
    unknown = requested - EXISTENCE_TEMPLATES
    if unknown:
        raise ValueError(
            f"Unknown or non-existence-based templates: {unknown}. "
            f"Supported in template mode: {sorted(EXISTENCE_TEMPLATES)}"
        )
    return requested


def unique_activities(constraints: list[dict]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for c in constraints:
        for act in c["activities"]:
            if act not in seen:
                seen.add(act)
                ordered.append(act)
    return ordered


# ── Code generation ────────────────────────────────────────────────────────────

def generate_module(dataset: str, constraints: list[dict], mode_desc: str) -> str:
    activities = unique_activities(constraints)
    counts = Counter(c["template"] for c in constraints)

    constraints_json = json.dumps(constraints, indent=4)
    activities_repr = repr(activities)
    counts_str = ", ".join(f"{t}={n}" for t, n in sorted(counts.items()))

    lines = []
    w = lines.append

    w(f'"""')
    w(f'Auto-generated LTN constraints from DECLARE rules.')
    w(f'')
    w(f'Dataset:   {dataset}')
    w(f'Mode:      {mode_desc}')
    w(f'Total:     {len(constraints)} constraints  ({counts_str})')
    w(f'')
    w(f'Usage in training script')
    w(f'------------------------')
    w(f'from data.rules.{dataset}_ltn_constraints import (')
    w(f'    CONSTRAINTS, make_predicates, make_constants,')
    w(f'    compute_level1_features, build_level2_formulas,')
    w(f')')
    w(f'')
    w(f'# Level 3 predicates (define once before the training loop)')
    w(f'predicates = make_predicates(activity_col_start=0, seq_len=40)')
    w(f'constants  = make_constants(activity_vocab)   # dict: name -> int code')
    w(f'')
    w(f'# Level 1 — augment x before feeding to the model')
    w(f'feats    = compute_level1_features(x, activity_vocab, activity_col_start=0, seq_len=40)')
    w(f'x_concat = torch.cat([x, feats.repeat_interleave(seq_len, dim=1)], dim=1)')
    w(f'')
    w(f'# Level 2 — add to SatAgg loss')
    w(f'formulas += build_level2_formulas(x_var, predicates, constants,')
    w(f'                                   Forall, Implies, Not, And, Or,')
    w(f'                                   activity_col_start=0, seq_len=40)')
    w(f'"""')
    w(f'')
    w(f'import torch')
    w(f'')
    w(f'# ── Constraint metadata ──────────────────────────────────────────────────────')
    w(f'CONSTRAINTS: list[dict] = {constraints_json}')
    w(f'')
    w(f'# Ordered list of all unique activities referenced by the constraints above')
    w(f'UNIQUE_ACTIVITIES: list[str] = {activities_repr}')
    w(f'')
    w(f'')
    w(f'# ── Level 3: shared predicates (architecture) ────────────────────────────────')
    w(f'def make_predicates(activity_col_start: int, seq_len: int) -> dict:')
    w(f'    import ltn')
    w(f'    s, e = activity_col_start, activity_col_start + seq_len')
    w(f'    HasAct = ltn.Predicate(func=lambda x, act: (')
    w(f'        x[:, s:e] == act[0].item()')
    w(f'    ).any(dim=1).float())')
    w(f'    IsFirst = ltn.Predicate(func=lambda x, act: (')
    w(f'        x[:, s] == act[0].item()')
    w(f'    ).float())')
    w(f'    ExactlyOnce = ltn.Predicate(func=lambda x, act: (')
    w(f'        (x[:, s:e] == act[0].item()).sum(dim=1) == 1')
    w(f'    ).float())')
    w(f'    return {{"HasAct": HasAct, "IsFirst": IsFirst, "ExactlyOnce": ExactlyOnce}}')
    w(f'')
    w(f'')
    w(f'# ── Level 3: activity constants ──────────────────────────────────────────────')
    w(f'def make_constants(activity_vocab: dict) -> dict:')
    w(f'    import ltn')
    w(f'    return {{')
    w(f'        name: ltn.Constant(torch.tensor([code]))')
    w(f'        for name, code in activity_vocab.items()')
    w(f'        if name in UNIQUE_ACTIVITIES')
    w(f'    }}')
    w(f'')
    w(f'')
    w(f'# ── Level 1: input feature computation ───────────────────────────────────────')
    w(f'def compute_level1_features(')
    w(f'    x: torch.Tensor,')
    w(f'    activity_vocab: dict,')
    w(f'    activity_col_start: int,')
    w(f'    seq_len: int,')
    w(f') -> torch.Tensor:')
    w(f'    """')
    w(f'    Returns a float tensor of shape (batch_size, N_active_constraints).')
    w(f'    1.0 = constraint satisfied by the prefix, 0.0 = violated.')
    w(f'    Constraints whose activities are absent from activity_vocab are skipped.')
    w(f'    """')
    w(f'    s, e = activity_col_start, activity_col_start + seq_len')
    w(f'    act = x[:, s:e].long()')
    w(f'    B, T = act.size()')
    w(f'    device = act.device')
    w(f'    features: list[torch.Tensor] = []')
    w(f'')
    w(f'    for c in CONSTRAINTS:')
    w(f'        t    = c["template"]')
    w(f'        acts = c["activities"]')
    w(f'')
    w(f'        # ── Unary ──────────────────────────────────────────────────────────')
    w(f'        if t in ("existence", "init", "exactly_one", "absence"):')
    w(f'            code = activity_vocab.get(acts[0])')
    w(f'            if code is None:')
    w(f'                continue')
    w(f'            if t == "existence":')
    w(f'                features.append((act == code).any(dim=1).float())')
    w(f'            elif t == "init":')
    w(f'                features.append((act[:, 0] == code).float())')
    w(f'            elif t == "exactly_one":')
    w(f'                features.append(((act == code).sum(dim=1) == 1).float())')
    w(f'            elif t == "absence":')
    w(f'                features.append((~(act == code).any(dim=1)).float())')
    w(f'')
    w(f'        # ── Binary coexistence / exclusion ─────────────────────────────────')
    w(f'        elif t in ("responded_existence", "coexistence", "noncoexistence"):')
    w(f'            code_a = activity_vocab.get(acts[0])')
    w(f'            code_b = activity_vocab.get(acts[1])')
    w(f'            if code_a is None or code_b is None:')
    w(f'                continue')
    w(f'            has_a = (act == code_a).any(dim=1)')
    w(f'            has_b = (act == code_b).any(dim=1)')
    w(f'            if t == "responded_existence":')
    w(f'                features.append((~has_a | has_b).float())')
    w(f'            elif t == "coexistence":')
    w(f'                features.append((has_a == has_b).float())')
    w(f'            elif t == "noncoexistence":')
    w(f'                features.append((~(has_a & has_b)).float())')
    w(f'')
    w(f'        # ── Precedence ─────────────────────────────────────────────────────')
    w(f'        elif t == "precedence":')
    w(f'            code_a = activity_vocab.get(acts[0])')
    w(f'            code_b = activity_vocab.get(acts[1])')
    w(f'            if code_a is None or code_b is None:')
    w(f'                continue')
    w(f'            _INF = T')
    w(f'            _idx = torch.arange(T, device=device).unsqueeze(0)')
    w(f'            first_a = torch.where(act == code_a, _idx,')
    w(f'                                  torch.tensor(_INF, device=device)).min(1).values')
    w(f'            first_b = torch.where(act == code_b, _idx,')
    w(f'                                  torch.tensor(_INF, device=device)).min(1).values')
    w(f'            violated = (first_b < _INF) & (first_b < first_a)')
    w(f'            features.append((~violated).float())')
    w(f'')
    w(f'        # ── Alternating response ───────────────────────────────────────────')
    w(f'        elif t == "altresponse":')
    w(f'            code_a = activity_vocab.get(acts[0])')
    w(f'            code_b = activity_vocab.get(acts[1])')
    w(f'            if code_a is None or code_b is None:')
    w(f'                continue')
    w(f'            pos_a = (act == code_a)')
    w(f'            pos_b = (act == code_b)')
    w(f'            s = pos_a.long().cumsum(dim=1) - pos_b.long().cumsum(dim=1)')
    w(f'            s_before = torch.cat([torch.zeros(B, 1, dtype=s.dtype, device=device), s[:, :-1]], dim=1)')
    w(f'            violated = (pos_a & (s_before > 0)).any(dim=1)')
    w(f'            features.append((~violated).float())')
    w(f'')
    w(f'        # ── Alternating precedence ─────────────────────────────────────────')
    w(f'        elif t == "altprecedence":')
    w(f'            code_a = activity_vocab.get(acts[0])')
    w(f'            code_b = activity_vocab.get(acts[1])')
    w(f'            if code_a is None or code_b is None:')
    w(f'                continue')
    w(f'            pos_a = (act == code_a)')
    w(f'            pos_b = (act == code_b)')
    w(f'            s = pos_a.long().cumsum(dim=1) - pos_b.long().cumsum(dim=1)')
    w(f'            s_before = torch.cat([torch.zeros(B, 1, dtype=s.dtype, device=device), s[:, :-1]], dim=1)')
    w(f'            violated = (pos_b & (s_before <= 0)).any(dim=1)')
    w(f'            features.append((~violated).float())')
    w(f'')
    w(f'        # ── Alternating succession (altprecedence AND altresponse) ──────────')
    w(f'        elif t == "altsuccession":')
    w(f'            code_a = activity_vocab.get(acts[0])')
    w(f'            code_b = activity_vocab.get(acts[1])')
    w(f'            if code_a is None or code_b is None:')
    w(f'                continue')
    w(f'            pos_a = (act == code_a)')
    w(f'            pos_b = (act == code_b)')
    w(f'            s = pos_a.long().cumsum(dim=1) - pos_b.long().cumsum(dim=1)')
    w(f'            s_before = torch.cat([torch.zeros(B, 1, dtype=s.dtype, device=device), s[:, :-1]], dim=1)')
    w(f'            vr = (pos_a & (s_before > 0)).any(dim=1)')
    w(f'            vp = (pos_b & (s_before <= 0)).any(dim=1)')
    w(f'            features.append((~(vr | vp)).float())')
    w(f'')
    w(f'        # ── Chain constraints (immediate succession) ───────────────────────')
    w(f'        elif t in ("chainresponse", "chainprecedence", "chainsuccession"):')
    w(f'            code_a = activity_vocab.get(acts[0])')
    w(f'            code_b = activity_vocab.get(acts[1])')
    w(f'            if code_a is None or code_b is None:')
    w(f'                continue')
    w(f'            if t == "chainresponse":')
    w(f'                # violated if A appears but next position is not B')
    w(f'                is_a    = act[:, :-1] == code_a')
    w(f'                next_nb = act[:, 1:]  != code_b')
    w(f'                violated = (is_a & next_nb).any(dim=1)')
    w(f'            elif t == "chainprecedence":')
    w(f'                # violated if B appears without A immediately before it')
    w(f'                is_b    = act[:, 1:]  == code_b')
    w(f'                prev_na = act[:, :-1] != code_a')
    w(f'                violated = (is_b & prev_na).any(dim=1) | (act[:, 0] == code_b)')
    w(f'            else:  # chainsuccession')
    w(f'                is_a_r  = act[:, :-1] == code_a')
    w(f'                next_nb = act[:, 1:]  != code_b')
    w(f'                viol_r  = (is_a_r & next_nb).any(dim=1)')
    w(f'                is_b_p  = act[:, 1:]  == code_b')
    w(f'                prev_na = act[:, :-1] != code_a')
    w(f'                viol_p  = (is_b_p & prev_na).any(dim=1) | (act[:, 0] == code_b)')
    w(f'                violated = viol_r | viol_p')
    w(f'            features.append((~violated).float())')
    w(f'')
    w(f'    if not features:')
    w(f'        return torch.zeros(x.size(0), 0, device=x.device)')
    w(f'    return torch.stack(features, dim=1)')
    w(f'')
    w(f'')
    w(f'# ── Level 2: LTN loss formulas (existence-based templates only) ──────────────')
    w(f'def build_level2_formulas(')
    w(f'    x_var, predicates, constants,')
    w(f'    Forall, Implies, Not, And, Or,')
    w(f'    activity_col_start: int = 0,')
    w(f'    seq_len: int = 40,')
    w(f') -> list:')
    w(f'    HasAct      = predicates["HasAct"]')
    w(f'    IsFirst     = predicates["IsFirst"]')
    w(f'    ExactlyOnce = predicates["ExactlyOnce"]')
    w(f'    s, e = activity_col_start, activity_col_start + seq_len')
    w(f'    formulas: list = []')
    w(f'    for c in CONSTRAINTS:')
    w(f'        t    = c["template"]')
    w(f'        acts = c["activities"]')
    w(f'        if t == "existence":')
    w(f'            const_a = constants.get(acts[0])')
    w(f'            if const_a is None: continue')
    w(f'            formulas.append(Forall(x_var, HasAct(x_var, const_a)))')
    w(f'        elif t == "exactly_one":')
    w(f'            const_a = constants.get(acts[0])')
    w(f'            if const_a is None: continue')
    w(f'            formulas.append(Forall(x_var, ExactlyOnce(x_var, const_a)))')
    w(f'        elif t == "init":')
    w(f'            const_a = constants.get(acts[0])')
    w(f'            if const_a is None: continue')
    w(f'            formulas.append(Forall(x_var, IsFirst(x_var, const_a)))')
    w(f'        elif t == "responded_existence":')
    w(f'            const_a = constants.get(acts[0])')
    w(f'            const_b = constants.get(acts[1])')
    w(f'            if const_a is None or const_b is None: continue')
    w(f'            _ca = const_a.value[0].item()')
    w(f'            formulas.append(Forall(x_var, HasAct(x_var, const_b),')
    w(f'                cond_vars=[x_var],')
    w(f'                cond_fn=lambda xv, _s=s, _e=e, _c=_ca: (xv.value[:, _s:_e] == _c).any(dim=1),')
    w(f'            ))')
    w(f'        elif t == "coexistence":')
    w(f'            const_a = constants.get(acts[0])')
    w(f'            const_b = constants.get(acts[1])')
    w(f'            if const_a is None or const_b is None: continue')
    w(f'            formulas.append(Forall(x_var, And(')
    w(f'                Implies(HasAct(x_var, const_a), HasAct(x_var, const_b)),')
    w(f'                Implies(HasAct(x_var, const_b), HasAct(x_var, const_a)),')
    w(f'            )))')
    w(f'    return formulas')
    w(f'')

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    if args.discriminability_csv:
        constraints = load_by_discriminability(
            args.dataset, args.discriminability_csv, args.min_net
        )
        mode_desc = (
            f"discriminability filter  "
            f"(csv={Path(args.discriminability_csv).name}, min_net={args.min_net})"
        )
    else:
        templates = resolve_templates(args.templates)
        constraints = load_by_template(
            args.dataset, templates, args.min_confidence, args.top_k
        )
        mode_desc = f"template filter  (templates={sorted(templates)}, min_conf={args.min_confidence})"

    if not constraints:
        print("No constraints matched the given filters. Nothing written.")
        return

    counts = Counter(c["template"] for c in constraints)
    print(f"Selected {len(constraints)} constraints:")
    for t, n in sorted(counts.items()):
        print(f"  {t:20s}  {n}")

    out_path = (
        Path(args.out)
        if args.out
        else ROOT / "data" / "rules" / f"{args.dataset}_ltn_constraints.py"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    module_src = generate_module(args.dataset, constraints, mode_desc)
    out_path.write_text(module_src)
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()
