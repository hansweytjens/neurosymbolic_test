"""
Convert DECLARE existence-based constraints to an importable LTN constraints module.

Output: data/rules/{dataset}_ltn_constraints.py

The generated module supports all three LTN integration levels:
  Level 1 (input):        compute_level1_features(x, activity_vocab, activity_col_start, seq_len)
  Level 2 (loss):         build_level2_formulas(x_var, predicates, constants, Forall, Implies, Not, And, Or)
  Level 3 (architecture): make_predicates(activity_col_start, seq_len)  /  make_constants(activity_vocab)

Usage:
    python rules/convert_declare_to_ltn.py --dataset bpi12
    python rules/convert_declare_to_ltn.py --dataset bpi12 --templates responded_existence coexistence
    python rules/convert_declare_to_ltn.py --dataset bpi12 --min-confidence 0.8 --top-k 20
"""

import argparse
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


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert DECLARE rules to LTN constraints module"
    )
    parser.add_argument(
        "--dataset", type=str, default="bpi12",
        help="Dataset name (default: bpi12)",
    )
    parser.add_argument(
        "--templates", type=str, nargs="+", default=["all"],
        help=(
            "Constraint templates to include. "
            "Use 'all' for all existence-based templates, or list specific names. "
            f"Available existence-based templates: {sorted(EXISTENCE_TEMPLATES)}. "
            "Default: all"
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
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output file path (default: data/rules/{dataset}_ltn_constraints.py)",
    )
    return parser.parse_args()


def resolve_templates(templates_arg: list[str]) -> set[str]:
    if templates_arg == ["all"]:
        return EXISTENCE_TEMPLATES
    requested = {t.lower() for t in templates_arg}
    unknown = requested - EXISTENCE_TEMPLATES
    if unknown:
        raise ValueError(
            f"Unknown or non-existence-based templates: {unknown}. "
            f"Supported: {sorted(EXISTENCE_TEMPLATES)}"
        )
    return requested


def load_and_filter(dataset: str, templates: set[str], min_confidence: float, top_k: int | None):
    path = ROOT / "data" / "rules" / f"{dataset}_declare.json"
    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    all_constraints = data["constraints"]

    # Filter by template and confidence
    filtered = [
        c for c in all_constraints
        if c["template"] in templates and c["confidence"] >= min_confidence
    ]

    # Apply top-k per template
    if top_k is not None:
        by_template: dict[str, list] = {}
        for c in filtered:
            by_template.setdefault(c["template"], []).append(c)
        filtered = []
        for template_constraints in by_template.values():
            ranked = sorted(template_constraints, key=lambda c: -c["confidence"])
            filtered.extend(ranked[:top_k])

    return filtered


def unique_activities(constraints: list[dict]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for c in constraints:
        for act in c["activities"]:
            if act not in seen:
                seen.add(act)
                ordered.append(act)
    return ordered


def generate_module(dataset: str, constraints: list[dict], templates: set[str]) -> str:
    activities = unique_activities(constraints)
    counts = Counter(c["template"] for c in constraints)

    constraints_json = json.dumps(constraints, indent=4)
    activities_repr = repr(activities)
    templates_repr = repr(sorted(templates))
    counts_str = ", ".join(f"{t}={n}" for t, n in sorted(counts.items()))

    lines = []
    w = lines.append  # shorthand

    w(f'"""')
    w(f'Auto-generated LTN constraints from DECLARE rules.')
    w(f'')
    w(f'Dataset:   {dataset}')
    w(f'Templates: {templates_repr}')
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
    w(f'    """')
    w(f'    Returns a dict of ltn.Predicate objects keyed by name.')
    w(f'    All predicates operate on the raw (flattened) input tensor x.')
    w(f'')
    w(f'    activity_col_start : column index where concept:name begins in x')
    w(f'    seq_len            : number of timesteps (= max prefix length)')
    w(f'')
    w(f'    Predicates')
    w(f'    ----------')
    w(f'    HasAct(x, act)      1 if act occurs anywhere in the prefix')
    w(f'    IsFirst(x, act)     1 if act is the very first event')
    w(f'    ExactlyOnce(x, act) 1 if act appears exactly once')
    w(f'    """')
    w(f'    import ltn')
    w(f'    s, e = activity_col_start, activity_col_start + seq_len')
    w(f'')
    w(f'    HasAct = ltn.Predicate(func=lambda x, act: (')
    w(f'        x[:, s:e] == act[0].item()')
    w(f'    ).any(dim=1).float())')
    w(f'')
    w(f'    IsFirst = ltn.Predicate(func=lambda x, act: (')
    w(f'        x[:, s] == act[0].item()')
    w(f'    ).float())')
    w(f'')
    w(f'    ExactlyOnce = ltn.Predicate(func=lambda x, act: (')
    w(f'        (x[:, s:e] == act[0].item()).sum(dim=1) == 1')
    w(f'    ).float())')
    w(f'')
    w(f'    return {{"HasAct": HasAct, "IsFirst": IsFirst, "ExactlyOnce": ExactlyOnce}}')
    w(f'')
    w(f'')
    w(f'# ── Level 3: activity constants ──────────────────────────────────────────────')
    w(f'def make_constants(activity_vocab: dict) -> dict:')
    w(f'    """')
    w(f'    activity_vocab : {{activity_name: integer_code, ...}}')
    w(f'                     (the mapping produced by pd.Categorical.cat.codes + 1')
    w(f'                      in preprocess_{dataset}.py)')
    w(f'')
    w(f'    Returns {{activity_name: ltn.Constant}} for every activity in')
    w(f'    UNIQUE_ACTIVITIES that is present in activity_vocab.')
    w(f'    Missing activities are silently skipped — affected constraints')
    w(f'    will be omitted from build_level2_formulas.')
    w(f'    """')
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
    w(f'    Compute a binary satisfaction score for each constraint.')
    w(f'')
    w(f'    Returns a float tensor of shape (batch_size, N_active_constraints)')
    w(f'    where 1.0 means the constraint is satisfied by the current prefix')
    w(f'    and 0.0 means it is violated.')
    w(f'')
    w(f'    Constraints whose activities are absent from activity_vocab are')
    w(f'    silently skipped (they do not appear in the output tensor).')
    w(f'')
    w(f'    Typical usage (mirrors the pattern from main_bpi12.py):')
    w(f'')
    w(f'        feats    = compute_level1_features(x, activity_vocab, 0, 40).detach()')
    w(f'        x_concat = torch.cat([x, feats.repeat_interleave(seq_len, dim=1)], dim=1)')
    w(f'    """')
    w(f'    s, e = activity_col_start, activity_col_start + seq_len')
    w(f'    act = x[:, s:e]')
    w(f'    features: list[torch.Tensor] = []')
    w(f'')
    w(f'    for c in CONSTRAINTS:')
    w(f'        t    = c["template"]')
    w(f'        acts = c["activities"]')
    w(f'')
    w(f'        if t == "existence":')
    w(f'            code = activity_vocab.get(acts[0])')
    w(f'            if code is None:')
    w(f'                continue')
    w(f'            features.append((act == code).any(dim=1).float())')
    w(f'')
    w(f'        elif t == "exactly_one":')
    w(f'            code = activity_vocab.get(acts[0])')
    w(f'            if code is None:')
    w(f'                continue')
    w(f'            features.append(((act == code).sum(dim=1) == 1).float())')
    w(f'')
    w(f'        elif t == "init":')
    w(f'            code = activity_vocab.get(acts[0])')
    w(f'            if code is None:')
    w(f'                continue')
    w(f'            features.append((act[:, 0] == code).float())')
    w(f'')
    w(f'        elif t == "responded_existence":')
    w(f'            # satisfied when: NOT has_A  OR  has_B')
    w(f'            code_a = activity_vocab.get(acts[0])')
    w(f'            code_b = activity_vocab.get(acts[1])')
    w(f'            if code_a is None or code_b is None:')
    w(f'                continue')
    w(f'            has_a = (act == code_a).any(dim=1)')
    w(f'            has_b = (act == code_b).any(dim=1)')
    w(f'            features.append((~has_a | has_b).float())')
    w(f'')
    w(f'        elif t == "coexistence":')
    w(f'            # satisfied when both occur or neither occurs')
    w(f'            code_a = activity_vocab.get(acts[0])')
    w(f'            code_b = activity_vocab.get(acts[1])')
    w(f'            if code_a is None or code_b is None:')
    w(f'                continue')
    w(f'            has_a = (act == code_a).any(dim=1)')
    w(f'            has_b = (act == code_b).any(dim=1)')
    w(f'            features.append((has_a == has_b).float())')
    w(f'')
    w(f'    if not features:')
    w(f'        return torch.zeros(x.size(0), 0, device=x.device)')
    w(f'    return torch.stack(features, dim=1)')
    w(f'')
    w(f'')
    w(f'# ── Level 2: LTN loss formulas ───────────────────────────────────────────────')
    w(f'def build_level2_formulas(')
    w(f'    x_var,')
    w(f'    predicates: dict,')
    w(f'    constants: dict,')
    w(f'    Forall,')
    w(f'    Implies,')
    w(f'    Not,')
    w(f'    And,')
    w(f'    Or,')
    w(f'    activity_col_start: int = 0,')
    w(f'    seq_len: int = 40,')
    w(f') -> list:')
    w(f'    """')
    w(f'    Build LTN formulas for all active constraints.')
    w(f'')
    w(f'    Returns a list of formula objects ready to pass to SatAgg:')
    w(f'')
    w(f'        formulas  = build_level2_formulas(x_All, predicates, constants,')
    w(f'                                           Forall, Implies, Not, And, Or,')
    w(f'                                           activity_col_start=0, seq_len=40)')
    w(f'        sat_agg   = SatAgg(*base_formulas, *formulas)')
    w(f'')
    w(f'    Constraints whose activities are missing from `constants` are skipped.')
    w(f'')
    w(f'    Formula shapes per template')
    w(f'    ---------------------------')
    w(f'    existence(A)             forall x: HasAct(x, A)')
    w(f'    exactly_one(A)           forall x: ExactlyOnce(x, A)')
    w(f'    init(A)                  forall x: IsFirst(x, A)')
    w(f'    responded_existence(A,B) forall x where has_A: HasAct(x, B)')
    w(f'    coexistence(A,B)         forall x: And(A->B, B->A)')
    w(f'    """')
    w(f'    HasAct      = predicates["HasAct"]')
    w(f'    IsFirst     = predicates["IsFirst"]')
    w(f'    ExactlyOnce = predicates["ExactlyOnce"]')
    w(f'    s, e = activity_col_start, activity_col_start + seq_len')
    w(f'    formulas: list = []')
    w(f'')
    w(f'    for c in CONSTRAINTS:')
    w(f'        t    = c["template"]')
    w(f'        acts = c["activities"]')
    w(f'')
    w(f'        if t == "existence":')
    w(f'            const_a = constants.get(acts[0])')
    w(f'            if const_a is None:')
    w(f'                continue')
    w(f'            formulas.append(Forall(x_var, HasAct(x_var, const_a)))')
    w(f'')
    w(f'        elif t == "exactly_one":')
    w(f'            const_a = constants.get(acts[0])')
    w(f'            if const_a is None:')
    w(f'                continue')
    w(f'            formulas.append(Forall(x_var, ExactlyOnce(x_var, const_a)))')
    w(f'')
    w(f'        elif t == "init":')
    w(f'            const_a = constants.get(acts[0])')
    w(f'            if const_a is None:')
    w(f'                continue')
    w(f'            formulas.append(Forall(x_var, IsFirst(x_var, const_a)))')
    w(f'')
    w(f'        elif t == "responded_existence":')
    w(f'            # forall x where A occurs: B must also occur')
    w(f'            # Uses cond_fn (same pattern as main_bpi12.py) to scope')
    w(f'            # the universal quantifier to antecedent-positive traces only.')
    w(f'            const_a = constants.get(acts[0])')
    w(f'            const_b = constants.get(acts[1])')
    w(f'            if const_a is None or const_b is None:')
    w(f'                continue')
    w(f'            # Default-arg binding avoids the loop-closure bug')
    w(f'            _ca = const_a.value[0].item()')
    w(f'            formulas.append(')
    w(f'                Forall(')
    w(f'                    x_var,')
    w(f'                    HasAct(x_var, const_b),')
    w(f'                    cond_vars=[x_var],')
    w(f'                    cond_fn=lambda xv, _s=s, _e=e, _c=_ca: (')
    w(f'                        xv.value[:, _s:_e] == _c')
    w(f'                    ).any(dim=1),')
    w(f'                )')
    w(f'            )')
    w(f'')
    w(f'        elif t == "coexistence":')
    w(f'            # forall x: HasAct(A) <-> HasAct(B)')
    w(f'            const_a = constants.get(acts[0])')
    w(f'            const_b = constants.get(acts[1])')
    w(f'            if const_a is None or const_b is None:')
    w(f'                continue')
    w(f'            formulas.append(')
    w(f'                Forall(')
    w(f'                    x_var,')
    w(f'                    And(')
    w(f'                        Implies(HasAct(x_var, const_a), HasAct(x_var, const_b)),')
    w(f'                        Implies(HasAct(x_var, const_b), HasAct(x_var, const_a)),')
    w(f'                    ),')
    w(f'                )')
    w(f'            )')
    w(f'')
    w(f'    return formulas')
    w(f'')

    return "\n".join(lines)


def main():
    args = get_args()
    templates = resolve_templates(args.templates)
    constraints = load_and_filter(args.dataset, templates, args.min_confidence, args.top_k)

    if not constraints:
        print("No constraints matched the given filters. Nothing written.")
        return

    counts = Counter(c["template"] for c in constraints)
    print(f"Selected {len(constraints)} constraints: " +
          ", ".join(f"{t}={n}" for t, n in sorted(counts.items())))

    out_path = (
        Path(args.out)
        if args.out
        else ROOT / "data" / "rules" / f"{args.dataset}_ltn_constraints.py"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    module_src = generate_module(args.dataset, constraints, templates)
    out_path.write_text(module_src)
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()
