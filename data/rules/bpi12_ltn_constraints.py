"""
Auto-generated LTN constraints from DECLARE rules.

Dataset:   bpi12
Templates: ['coexistence', 'exactly_one', 'existence', 'init', 'responded_existence']
Total:     9 constraints  (coexistence=2, exactly_one=2, existence=2, init=1, responded_existence=2)

Usage in training script
------------------------
from data.rules.bpi12_ltn_constraints import (
    CONSTRAINTS, make_predicates, make_constants,
    compute_level1_features, build_level2_formulas,
)

# Level 3 predicates (define once before the training loop)
predicates = make_predicates(activity_col_start=0, seq_len=40)
constants  = make_constants(activity_vocab)   # dict: name -> int code

# Level 1 — augment x before feeding to the model
feats    = compute_level1_features(x, activity_vocab, activity_col_start=0, seq_len=40)
x_concat = torch.cat([x, feats.repeat_interleave(seq_len, dim=1)], dim=1)

# Level 2 — add to SatAgg loss
formulas += build_level2_formulas(x_var, predicates, constants,
                                   Forall, Implies, Not, And, Or,
                                   activity_col_start=0, seq_len=40)
"""

import torch

# ── Constraint metadata ──────────────────────────────────────────────────────
CONSTRAINTS: list[dict] = [
    {
        "id": "c0001",
        "template": "exactly_one",
        "arity": "unary",
        "activities": [
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "'A_SUBMITTED-COMPLETE' must occur exactly once"
    },
    {
        "id": "c0002",
        "template": "exactly_one",
        "arity": "unary",
        "activities": [
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "'A_PARTLYSUBMITTED-COMPLETE' must occur exactly once"
    },
    {
        "id": "c0004",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_PARTLYSUBMITTED-COMPLETE",
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "If 'A_PARTLYSUBMITTED-COMPLETE' occurs, 'A_SUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0005",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_SUBMITTED-COMPLETE",
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "If 'A_SUBMITTED-COMPLETE' occurs, 'A_PARTLYSUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0006",
        "template": "existence",
        "arity": "unary",
        "activities": [
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "'A_PARTLYSUBMITTED-COMPLETE' must occur at least once"
    },
    {
        "id": "c0007",
        "template": "existence",
        "arity": "unary",
        "activities": [
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "'A_SUBMITTED-COMPLETE' must occur at least once"
    },
    {
        "id": "c0009",
        "template": "init",
        "arity": "unary",
        "activities": [
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "'A_SUBMITTED-COMPLETE' must be the first activity in every trace"
    },
    {
        "id": "c0029",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_SUBMITTED-COMPLETE",
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "'A_SUBMITTED-COMPLETE' and 'A_PARTLYSUBMITTED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0030",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_PARTLYSUBMITTED-COMPLETE",
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "'A_PARTLYSUBMITTED-COMPLETE' and 'A_SUBMITTED-COMPLETE' must both occur, or neither occurs"
    }
]

# Ordered list of all unique activities referenced by the constraints above
UNIQUE_ACTIVITIES: list[str] = ['A_SUBMITTED-COMPLETE', 'A_PARTLYSUBMITTED-COMPLETE']


# ── Level 3: shared predicates (architecture) ────────────────────────────────
def make_predicates(activity_col_start: int, seq_len: int) -> dict:
    """
    Returns a dict of ltn.Predicate objects keyed by name.
    All predicates operate on the raw (flattened) input tensor x.

    activity_col_start : column index where concept:name begins in x
    seq_len            : number of timesteps (= max prefix length)

    Predicates
    ----------
    HasAct(x, act)      1 if act occurs anywhere in the prefix
    IsFirst(x, act)     1 if act is the very first event
    ExactlyOnce(x, act) 1 if act appears exactly once
    """
    import ltn
    s, e = activity_col_start, activity_col_start + seq_len

    HasAct = ltn.Predicate(func=lambda x, act: (
        x[:, s:e] == act[0].item()
    ).any(dim=1).float())

    IsFirst = ltn.Predicate(func=lambda x, act: (
        x[:, s] == act[0].item()
    ).float())

    ExactlyOnce = ltn.Predicate(func=lambda x, act: (
        (x[:, s:e] == act[0].item()).sum(dim=1) == 1
    ).float())

    return {"HasAct": HasAct, "IsFirst": IsFirst, "ExactlyOnce": ExactlyOnce}


# ── Level 3: activity constants ──────────────────────────────────────────────
def make_constants(activity_vocab: dict) -> dict:
    """
    activity_vocab : {activity_name: integer_code, ...}
                     (the mapping produced by pd.Categorical.cat.codes + 1
                      in preprocess_bpi12.py)

    Returns {activity_name: ltn.Constant} for every activity in
    UNIQUE_ACTIVITIES that is present in activity_vocab.
    Missing activities are silently skipped — affected constraints
    will be omitted from build_level2_formulas.
    """
    import ltn
    return {
        name: ltn.Constant(torch.tensor([code]))
        for name, code in activity_vocab.items()
        if name in UNIQUE_ACTIVITIES
    }


# ── Level 1: input feature computation ───────────────────────────────────────
def compute_level1_features(
    x: torch.Tensor,
    activity_vocab: dict,
    activity_col_start: int,
    seq_len: int,
) -> torch.Tensor:
    """
    Compute a binary satisfaction score for each constraint.

    Returns a float tensor of shape (batch_size, N_active_constraints)
    where 1.0 means the constraint is satisfied by the current prefix
    and 0.0 means it is violated.

    Constraints whose activities are absent from activity_vocab are
    silently skipped (they do not appear in the output tensor).

    Typical usage (mirrors the pattern from main_bpi12.py):

        feats    = compute_level1_features(x, activity_vocab, 0, 40).detach()
        x_concat = torch.cat([x, feats.repeat_interleave(seq_len, dim=1)], dim=1)
    """
    s, e = activity_col_start, activity_col_start + seq_len
    act = x[:, s:e]
    features: list[torch.Tensor] = []

    for c in CONSTRAINTS:
        t    = c["template"]
        acts = c["activities"]

        if t == "existence":
            code = activity_vocab.get(acts[0])
            if code is None:
                continue
            features.append((act == code).any(dim=1).float())

        elif t == "exactly_one":
            code = activity_vocab.get(acts[0])
            if code is None:
                continue
            features.append(((act == code).sum(dim=1) == 1).float())

        elif t == "init":
            code = activity_vocab.get(acts[0])
            if code is None:
                continue
            features.append((act[:, 0] == code).float())

        elif t == "responded_existence":
            # satisfied when: NOT has_A  OR  has_B
            code_a = activity_vocab.get(acts[0])
            code_b = activity_vocab.get(acts[1])
            if code_a is None or code_b is None:
                continue
            has_a = (act == code_a).any(dim=1)
            has_b = (act == code_b).any(dim=1)
            features.append((~has_a | has_b).float())

        elif t == "coexistence":
            # satisfied when both occur or neither occurs
            code_a = activity_vocab.get(acts[0])
            code_b = activity_vocab.get(acts[1])
            if code_a is None or code_b is None:
                continue
            has_a = (act == code_a).any(dim=1)
            has_b = (act == code_b).any(dim=1)
            features.append((has_a == has_b).float())

    if not features:
        return torch.zeros(x.size(0), 0, device=x.device)
    return torch.stack(features, dim=1)


# ── Level 2: LTN loss formulas ───────────────────────────────────────────────
def build_level2_formulas(
    x_var,
    predicates: dict,
    constants: dict,
    Forall,
    Implies,
    Not,
    And,
    Or,
    activity_col_start: int = 0,
    seq_len: int = 40,
) -> list:
    """
    Build LTN formulas for all active constraints.

    Returns a list of formula objects ready to pass to SatAgg:

        formulas  = build_level2_formulas(x_All, predicates, constants,
                                           Forall, Implies, Not, And, Or,
                                           activity_col_start=0, seq_len=40)
        sat_agg   = SatAgg(*base_formulas, *formulas)

    Constraints whose activities are missing from `constants` are skipped.

    Formula shapes per template
    ---------------------------
    existence(A)             forall x: HasAct(x, A)
    exactly_one(A)           forall x: ExactlyOnce(x, A)
    init(A)                  forall x: IsFirst(x, A)
    responded_existence(A,B) forall x where has_A: HasAct(x, B)
    coexistence(A,B)         forall x: And(A->B, B->A)
    """
    HasAct      = predicates["HasAct"]
    IsFirst     = predicates["IsFirst"]
    ExactlyOnce = predicates["ExactlyOnce"]
    s, e = activity_col_start, activity_col_start + seq_len
    formulas: list = []

    for c in CONSTRAINTS:
        t    = c["template"]
        acts = c["activities"]

        if t == "existence":
            const_a = constants.get(acts[0])
            if const_a is None:
                continue
            formulas.append(Forall(x_var, HasAct(x_var, const_a)))

        elif t == "exactly_one":
            const_a = constants.get(acts[0])
            if const_a is None:
                continue
            formulas.append(Forall(x_var, ExactlyOnce(x_var, const_a)))

        elif t == "init":
            const_a = constants.get(acts[0])
            if const_a is None:
                continue
            formulas.append(Forall(x_var, IsFirst(x_var, const_a)))

        elif t == "responded_existence":
            # forall x where A occurs: B must also occur
            # Uses cond_fn (same pattern as main_bpi12.py) to scope
            # the universal quantifier to antecedent-positive traces only.
            const_a = constants.get(acts[0])
            const_b = constants.get(acts[1])
            if const_a is None or const_b is None:
                continue
            # Default-arg binding avoids the loop-closure bug
            _ca = const_a.value[0].item()
            formulas.append(
                Forall(
                    x_var,
                    HasAct(x_var, const_b),
                    cond_vars=[x_var],
                    cond_fn=lambda xv, _s=s, _e=e, _c=_ca: (
                        xv.value[:, _s:_e] == _c
                    ).any(dim=1),
                )
            )

        elif t == "coexistence":
            # forall x: HasAct(A) <-> HasAct(B)
            const_a = constants.get(acts[0])
            const_b = constants.get(acts[1])
            if const_a is None or const_b is None:
                continue
            formulas.append(
                Forall(
                    x_var,
                    And(
                        Implies(HasAct(x_var, const_a), HasAct(x_var, const_b)),
                        Implies(HasAct(x_var, const_b), HasAct(x_var, const_a)),
                    ),
                )
            )

    return formulas
