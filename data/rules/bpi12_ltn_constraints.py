"""
Auto-generated LTN constraints from DECLARE rules.

Dataset:   bpi12
Templates: ['coexistence', 'exactly_one', 'existence', 'init', 'responded_existence']
Total:     221 constraints  (coexistence=76, exactly_one=2, existence=2, init=1, responded_existence=140)

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
import ltn


# ── Constraint metadata ──────────────────────────────────────────────────────
CONSTRAINTS: list[dict] = [
    {
        "id": "c0001",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "O_SELECTED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4214,
        "description": "If 'A_ACCEPTED-COMPLETE' occurs, 'O_SELECTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0002",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SELECTED-COMPLETE' occurs, 'O_SENT-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0003",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4212,
        "confidence": 0.4212,
        "description": "If 'W_Nabellen offertes-COMPLETE' occurs, 'O_SENT-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0004",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "W_Completeren aanvraag-SCHEDULE"
        ],
        "support": 0.4207,
        "confidence": 0.4207,
        "description": "If 'W_Nabellen offertes-START' occurs, 'W_Completeren aanvraag-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0005",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "A_ACCEPTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SENT-COMPLETE' occurs, 'A_ACCEPTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0006",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "W_Nabellen offertes-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4212,
        "description": "If 'A_FINALIZED-COMPLETE' occurs, 'W_Nabellen offertes-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0007",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "A_FINALIZED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'W_Nabellen offertes-SCHEDULE' occurs, 'A_FINALIZED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0008",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "W_Nabellen offertes-SCHEDULE"
        ],
        "support": 0.4207,
        "confidence": 0.4207,
        "description": "If 'W_Nabellen offertes-START' occurs, 'W_Nabellen offertes-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0009",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "W_Nabellen offertes-START"
        ],
        "support": 0.4214,
        "confidence": 0.4207,
        "description": "If 'O_CREATED-COMPLETE' occurs, 'W_Nabellen offertes-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0010",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-SCHEDULE",
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 0.3551,
        "confidence": 0.3551,
        "description": "If 'W_Afhandelen leads-SCHEDULE' occurs, 'A_SUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0011",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "W_Nabellen offertes-COMPLETE"
        ],
        "support": 0.4207,
        "confidence": 0.4207,
        "description": "If 'W_Nabellen offertes-START' occurs, 'W_Nabellen offertes-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0012",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-COMPLETE",
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 0.5967,
        "confidence": 0.5967,
        "description": "If 'W_Completeren aanvraag-COMPLETE' occurs, 'A_PARTLYSUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0013",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "O_CREATED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'A_FINALIZED-COMPLETE' occurs, 'O_CREATED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0014",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'W_Nabellen offertes-SCHEDULE' occurs, 'O_SENT-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0015",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "W_Completeren aanvraag-START"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'A_FINALIZED-COMPLETE' occurs, 'W_Completeren aanvraag-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0016",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "W_Completeren aanvraag-SCHEDULE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SENT-COMPLETE' occurs, 'W_Completeren aanvraag-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0017",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_PREACCEPTED-COMPLETE",
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 0.5967,
        "confidence": 0.5967,
        "description": "If 'A_PREACCEPTED-COMPLETE' occurs, 'A_SUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0018",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "W_Nabellen offertes-SCHEDULE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SENT-COMPLETE' occurs, 'W_Nabellen offertes-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0019",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "W_Nabellen offertes-START"
        ],
        "support": 0.4214,
        "confidence": 0.4207,
        "description": "If 'O_SELECTED-COMPLETE' occurs, 'W_Nabellen offertes-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0020",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "W_Completeren aanvraag-START"
        ],
        "support": 0.4207,
        "confidence": 0.4207,
        "description": "If 'W_Nabellen offertes-START' occurs, 'W_Completeren aanvraag-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0021",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'A_FINALIZED-COMPLETE' occurs, 'A_PARTLYSUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0022",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'A_FINALIZED-COMPLETE' occurs, 'A_SUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0023",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 0.4207,
        "confidence": 0.4207,
        "description": "If 'W_Nabellen offertes-START' occurs, 'A_PARTLYSUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0024",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 0.4207,
        "confidence": 0.4207,
        "description": "If 'W_Nabellen offertes-START' occurs, 'A_SUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0025",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-START",
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 0.3551,
        "confidence": 0.3551,
        "description": "If 'W_Afhandelen leads-START' occurs, 'A_PARTLYSUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0026",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "W_Completeren aanvraag-START"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SENT-COMPLETE' occurs, 'W_Completeren aanvraag-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0027",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "A_FINALIZED-COMPLETE"
        ],
        "support": 0.4207,
        "confidence": 0.4207,
        "description": "If 'W_Nabellen offertes-START' occurs, 'A_FINALIZED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0028",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'A_FINALIZED-COMPLETE' occurs, 'O_SENT-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0029",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SENT-COMPLETE' occurs, 'A_PARTLYSUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0030",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "W_Nabellen offertes-SCHEDULE"
        ],
        "support": 0.4298,
        "confidence": 0.4214,
        "description": "If 'A_ACCEPTED-COMPLETE' occurs, 'W_Nabellen offertes-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0031",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SENT-COMPLETE' occurs, 'A_SUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0032",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-COMPLETE",
            "A_PREACCEPTED-COMPLETE"
        ],
        "support": 0.5967,
        "confidence": 0.5967,
        "description": "If 'W_Completeren aanvraag-COMPLETE' occurs, 'A_PREACCEPTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0033",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4207,
        "confidence": 0.4207,
        "description": "If 'W_Nabellen offertes-START' occurs, 'O_SENT-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0034",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-SCHEDULE",
            "W_Completeren aanvraag-START"
        ],
        "support": 0.5967,
        "confidence": 0.5966,
        "description": "If 'W_Completeren aanvraag-SCHEDULE' occurs, 'W_Completeren aanvraag-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0035",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "W_Nabellen offertes-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4212,
        "description": "If 'A_ACCEPTED-COMPLETE' occurs, 'W_Nabellen offertes-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0036",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "A_FINALIZED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SENT-COMPLETE' occurs, 'A_FINALIZED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0037",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "O_CREATED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4214,
        "description": "If 'A_ACCEPTED-COMPLETE' occurs, 'O_CREATED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0038",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-SCHEDULE",
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 0.5967,
        "confidence": 0.5967,
        "description": "If 'W_Completeren aanvraag-SCHEDULE' occurs, 'A_PARTLYSUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0039",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "W_Nabellen offertes-START"
        ],
        "support": 0.4214,
        "confidence": 0.4207,
        "description": "If 'A_FINALIZED-COMPLETE' occurs, 'W_Nabellen offertes-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0040",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "W_Completeren aanvraag-START"
        ],
        "support": 0.4298,
        "confidence": 0.4298,
        "description": "If 'A_ACCEPTED-COMPLETE' occurs, 'W_Completeren aanvraag-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0041",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-COMPLETE",
            "W_Completeren aanvraag-SCHEDULE"
        ],
        "support": 0.5967,
        "confidence": 0.5967,
        "description": "If 'W_Completeren aanvraag-COMPLETE' occurs, 'W_Completeren aanvraag-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0042",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4298,
        "description": "If 'A_ACCEPTED-COMPLETE' occurs, 'A_PARTLYSUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0043",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_DECLINED-COMPLETE",
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 0.559,
        "confidence": 0.559,
        "description": "If 'A_DECLINED-COMPLETE' occurs, 'A_PARTLYSUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0044",
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
        "id": "c0045",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "W_Nabellen offertes-START"
        ],
        "support": 0.4214,
        "confidence": 0.4207,
        "description": "If 'O_SENT-COMPLETE' occurs, 'W_Nabellen offertes-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0046",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-START",
            "W_Afhandelen leads-SCHEDULE"
        ],
        "support": 0.3551,
        "confidence": 0.3551,
        "description": "If 'W_Afhandelen leads-START' occurs, 'W_Afhandelen leads-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0047",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "W_Completeren aanvraag-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_CREATED-COMPLETE' occurs, 'W_Completeren aanvraag-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0048",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "O_CREATED-COMPLETE"
        ],
        "support": 0.4212,
        "confidence": 0.4212,
        "description": "If 'W_Nabellen offertes-COMPLETE' occurs, 'O_CREATED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0049",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "W_Nabellen offertes-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4212,
        "description": "If 'W_Nabellen offertes-SCHEDULE' occurs, 'W_Nabellen offertes-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0050",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-COMPLETE",
            "W_Afhandelen leads-SCHEDULE"
        ],
        "support": 0.3551,
        "confidence": 0.3551,
        "description": "If 'W_Afhandelen leads-COMPLETE' occurs, 'W_Afhandelen leads-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0051",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-SCHEDULE",
            "W_Afhandelen leads-START"
        ],
        "support": 0.3551,
        "confidence": 0.3551,
        "description": "If 'W_Afhandelen leads-SCHEDULE' occurs, 'W_Afhandelen leads-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0052",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "O_CREATED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'W_Nabellen offertes-SCHEDULE' occurs, 'O_CREATED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0053",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-COMPLETE",
            "W_Afhandelen leads-START"
        ],
        "support": 0.3551,
        "confidence": 0.3551,
        "description": "If 'W_Afhandelen leads-COMPLETE' occurs, 'W_Afhandelen leads-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0054",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-COMPLETE",
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 0.5967,
        "confidence": 0.5967,
        "description": "If 'W_Completeren aanvraag-COMPLETE' occurs, 'A_SUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0055",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "O_SELECTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_CREATED-COMPLETE' occurs, 'O_SELECTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0056",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-SCHEDULE",
            "A_PREACCEPTED-COMPLETE"
        ],
        "support": 0.5967,
        "confidence": 0.5967,
        "description": "If 'W_Completeren aanvraag-SCHEDULE' occurs, 'A_PREACCEPTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0057",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "W_Completeren aanvraag-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SELECTED-COMPLETE' occurs, 'W_Completeren aanvraag-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0058",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "W_Completeren aanvraag-START"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'W_Nabellen offertes-SCHEDULE' occurs, 'W_Completeren aanvraag-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0059",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-START",
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 0.5966,
        "confidence": 0.5966,
        "description": "If 'W_Completeren aanvraag-START' occurs, 'A_PARTLYSUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0060",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-START",
            "W_Completeren aanvraag-COMPLETE"
        ],
        "support": 0.5966,
        "confidence": 0.5966,
        "description": "If 'W_Completeren aanvraag-START' occurs, 'W_Completeren aanvraag-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0061",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "W_Completeren aanvraag-COMPLETE"
        ],
        "support": 0.4212,
        "confidence": 0.4212,
        "description": "If 'W_Nabellen offertes-COMPLETE' occurs, 'W_Completeren aanvraag-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0062",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-START",
            "W_Afhandelen leads-COMPLETE"
        ],
        "support": 0.3551,
        "confidence": 0.3551,
        "description": "If 'W_Afhandelen leads-START' occurs, 'W_Afhandelen leads-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0063",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-SCHEDULE",
            "W_Afhandelen leads-COMPLETE"
        ],
        "support": 0.3551,
        "confidence": 0.3551,
        "description": "If 'W_Afhandelen leads-SCHEDULE' occurs, 'W_Afhandelen leads-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0064",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'W_Nabellen offertes-SCHEDULE' occurs, 'A_PARTLYSUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0065",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "W_Completeren aanvraag-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'W_Nabellen offertes-SCHEDULE' occurs, 'W_Completeren aanvraag-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0066",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "W_Nabellen offertes-START"
        ],
        "support": 0.4298,
        "confidence": 0.4207,
        "description": "If 'A_ACCEPTED-COMPLETE' occurs, 'W_Nabellen offertes-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0067",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "A_PREACCEPTED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4298,
        "description": "If 'A_ACCEPTED-COMPLETE' occurs, 'A_PREACCEPTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0068",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "O_SELECTED-COMPLETE"
        ],
        "support": 0.4212,
        "confidence": 0.4212,
        "description": "If 'W_Nabellen offertes-COMPLETE' occurs, 'O_SELECTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0069",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-START",
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 0.3551,
        "confidence": 0.3551,
        "description": "If 'W_Afhandelen leads-START' occurs, 'A_SUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0070",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "O_SELECTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'W_Nabellen offertes-SCHEDULE' occurs, 'O_SELECTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0071",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-SCHEDULE",
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 0.3551,
        "confidence": 0.3551,
        "description": "If 'W_Afhandelen leads-SCHEDULE' occurs, 'A_PARTLYSUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0072",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_PREACCEPTED-COMPLETE",
            "W_Completeren aanvraag-START"
        ],
        "support": 0.5967,
        "confidence": 0.5966,
        "description": "If 'A_PREACCEPTED-COMPLETE' occurs, 'W_Completeren aanvraag-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0073",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "A_PREACCEPTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_CREATED-COMPLETE' occurs, 'A_PREACCEPTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0074",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-COMPLETE",
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 0.3551,
        "confidence": 0.3551,
        "description": "If 'W_Afhandelen leads-COMPLETE' occurs, 'A_PARTLYSUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0075",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-COMPLETE",
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 0.3551,
        "confidence": 0.3551,
        "description": "If 'W_Afhandelen leads-COMPLETE' occurs, 'A_SUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0076",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "A_ACCEPTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_CREATED-COMPLETE' occurs, 'A_ACCEPTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0077",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "W_Completeren aanvraag-SCHEDULE"
        ],
        "support": 0.4298,
        "confidence": 0.4298,
        "description": "If 'A_ACCEPTED-COMPLETE' occurs, 'W_Completeren aanvraag-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0078",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_PREACCEPTED-COMPLETE",
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 0.5967,
        "confidence": 0.5967,
        "description": "If 'A_PREACCEPTED-COMPLETE' occurs, 'A_PARTLYSUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0079",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_PREACCEPTED-COMPLETE",
            "W_Completeren aanvraag-COMPLETE"
        ],
        "support": 0.5967,
        "confidence": 0.5967,
        "description": "If 'A_PREACCEPTED-COMPLETE' occurs, 'W_Completeren aanvraag-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0080",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "O_CREATED-COMPLETE"
        ],
        "support": 0.4207,
        "confidence": 0.4207,
        "description": "If 'W_Nabellen offertes-START' occurs, 'O_CREATED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0081",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "A_PREACCEPTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SELECTED-COMPLETE' occurs, 'A_PREACCEPTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0082",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-START",
            "A_PREACCEPTED-COMPLETE"
        ],
        "support": 0.5966,
        "confidence": 0.5966,
        "description": "If 'W_Completeren aanvraag-START' occurs, 'A_PREACCEPTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0083",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "W_Nabellen offertes-START"
        ],
        "support": 0.4212,
        "confidence": 0.4207,
        "description": "If 'W_Nabellen offertes-COMPLETE' occurs, 'W_Nabellen offertes-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0084",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "A_PREACCEPTED-COMPLETE"
        ],
        "support": 0.4212,
        "confidence": 0.4212,
        "description": "If 'W_Nabellen offertes-COMPLETE' occurs, 'A_PREACCEPTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0085",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "A_ACCEPTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SELECTED-COMPLETE' occurs, 'A_ACCEPTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0086",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "W_Nabellen offertes-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4212,
        "description": "If 'O_SENT-COMPLETE' occurs, 'W_Nabellen offertes-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0087",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "W_Completeren aanvraag-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'A_FINALIZED-COMPLETE' occurs, 'W_Completeren aanvraag-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0088",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "W_Completeren aanvraag-SCHEDULE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_CREATED-COMPLETE' occurs, 'W_Completeren aanvraag-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0089",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "A_ACCEPTED-COMPLETE"
        ],
        "support": 0.4212,
        "confidence": 0.4212,
        "description": "If 'W_Nabellen offertes-COMPLETE' occurs, 'A_ACCEPTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0090",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "W_Nabellen offertes-SCHEDULE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_CREATED-COMPLETE' occurs, 'W_Nabellen offertes-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0091",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-SCHEDULE",
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 0.5967,
        "confidence": 0.5967,
        "description": "If 'W_Completeren aanvraag-SCHEDULE' occurs, 'A_SUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0092",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "W_Nabellen offertes-START"
        ],
        "support": 0.4214,
        "confidence": 0.4207,
        "description": "If 'W_Nabellen offertes-SCHEDULE' occurs, 'W_Nabellen offertes-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0093",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "A_PREACCEPTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'W_Nabellen offertes-SCHEDULE' occurs, 'A_PREACCEPTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0094",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "W_Completeren aanvraag-COMPLETE"
        ],
        "support": 0.4207,
        "confidence": 0.4207,
        "description": "If 'W_Nabellen offertes-START' occurs, 'W_Completeren aanvraag-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0095",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "O_CREATED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SENT-COMPLETE' occurs, 'O_CREATED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0096",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "W_Nabellen offertes-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4212,
        "description": "If 'O_CREATED-COMPLETE' occurs, 'W_Nabellen offertes-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0097",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "A_ACCEPTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'W_Nabellen offertes-SCHEDULE' occurs, 'A_ACCEPTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0098",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "O_SELECTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'A_FINALIZED-COMPLETE' occurs, 'O_SELECTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0099",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "W_Completeren aanvraag-SCHEDULE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SELECTED-COMPLETE' occurs, 'W_Completeren aanvraag-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0100",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-START",
            "W_Completeren aanvraag-SCHEDULE"
        ],
        "support": 0.5966,
        "confidence": 0.5966,
        "description": "If 'W_Completeren aanvraag-START' occurs, 'W_Completeren aanvraag-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0101",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "W_Nabellen offertes-SCHEDULE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SELECTED-COMPLETE' occurs, 'W_Nabellen offertes-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0102",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "W_Completeren aanvraag-SCHEDULE"
        ],
        "support": 0.4212,
        "confidence": 0.4212,
        "description": "If 'W_Nabellen offertes-COMPLETE' occurs, 'W_Completeren aanvraag-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0103",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4298,
        "description": "If 'A_ACCEPTED-COMPLETE' occurs, 'A_SUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0104",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "W_Nabellen offertes-SCHEDULE"
        ],
        "support": 0.4212,
        "confidence": 0.4212,
        "description": "If 'W_Nabellen offertes-COMPLETE' occurs, 'W_Nabellen offertes-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0105",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "O_SELECTED-COMPLETE"
        ],
        "support": 0.4207,
        "confidence": 0.4207,
        "description": "If 'W_Nabellen offertes-START' occurs, 'O_SELECTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0106",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "W_Completeren aanvraag-START"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_CREATED-COMPLETE' occurs, 'W_Completeren aanvraag-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0107",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "W_Nabellen offertes-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4212,
        "description": "If 'O_SELECTED-COMPLETE' occurs, 'W_Nabellen offertes-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0108",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_DECLINED-COMPLETE",
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 0.559,
        "confidence": 0.559,
        "description": "If 'A_DECLINED-COMPLETE' occurs, 'A_SUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0109",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "W_Completeren aanvraag-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SENT-COMPLETE' occurs, 'W_Completeren aanvraag-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0110",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "W_Completeren aanvraag-SCHEDULE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'W_Nabellen offertes-SCHEDULE' occurs, 'W_Completeren aanvraag-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0111",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "O_CREATED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SELECTED-COMPLETE' occurs, 'O_CREATED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0112",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_CREATED-COMPLETE' occurs, 'A_PARTLYSUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0113",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "A_FINALIZED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4214,
        "description": "If 'A_ACCEPTED-COMPLETE' occurs, 'A_FINALIZED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0114",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_CREATED-COMPLETE' occurs, 'A_SUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0115",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "W_Completeren aanvraag-START"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SELECTED-COMPLETE' occurs, 'W_Completeren aanvraag-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0116",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "O_SELECTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SENT-COMPLETE' occurs, 'O_SELECTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0117",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "W_Completeren aanvraag-START"
        ],
        "support": 0.4212,
        "confidence": 0.4212,
        "description": "If 'W_Nabellen offertes-COMPLETE' occurs, 'W_Completeren aanvraag-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0118",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4214,
        "description": "If 'A_ACCEPTED-COMPLETE' occurs, 'O_SENT-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0119",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-SCHEDULE",
            "W_Completeren aanvraag-COMPLETE"
        ],
        "support": 0.5967,
        "confidence": 0.5967,
        "description": "If 'W_Completeren aanvraag-SCHEDULE' occurs, 'W_Completeren aanvraag-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0120",
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
        "id": "c0121",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "A_PREACCEPTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'A_FINALIZED-COMPLETE' occurs, 'A_PREACCEPTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0122",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SELECTED-COMPLETE' occurs, 'A_PARTLYSUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0123",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "A_FINALIZED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_CREATED-COMPLETE' occurs, 'A_FINALIZED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0124",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "A_ACCEPTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'A_FINALIZED-COMPLETE' occurs, 'A_ACCEPTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0125",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SELECTED-COMPLETE' occurs, 'A_SUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0126",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "A_PARTLYSUBMITTED-COMPLETE"
        ],
        "support": 0.4212,
        "confidence": 0.4212,
        "description": "If 'W_Nabellen offertes-COMPLETE' occurs, 'A_PARTLYSUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0127",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-START",
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 0.5966,
        "confidence": 0.5966,
        "description": "If 'W_Completeren aanvraag-START' occurs, 'A_SUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0128",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "A_PREACCEPTED-COMPLETE"
        ],
        "support": 0.4207,
        "confidence": 0.4207,
        "description": "If 'W_Nabellen offertes-START' occurs, 'A_PREACCEPTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0129",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 0.4212,
        "confidence": 0.4212,
        "description": "If 'W_Nabellen offertes-COMPLETE' occurs, 'A_SUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0130",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_CREATED-COMPLETE' occurs, 'O_SENT-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0131",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "A_ACCEPTED-COMPLETE"
        ],
        "support": 0.4207,
        "confidence": 0.4207,
        "description": "If 'W_Nabellen offertes-START' occurs, 'A_ACCEPTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0132",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "W_Completeren aanvraag-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4298,
        "description": "If 'A_ACCEPTED-COMPLETE' occurs, 'W_Completeren aanvraag-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0133",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_PREACCEPTED-COMPLETE",
            "W_Completeren aanvraag-SCHEDULE"
        ],
        "support": 0.5967,
        "confidence": 0.5967,
        "description": "If 'A_PREACCEPTED-COMPLETE' occurs, 'W_Completeren aanvraag-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0134",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'W_Nabellen offertes-SCHEDULE' occurs, 'A_SUBMITTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0135",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "A_FINALIZED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SELECTED-COMPLETE' occurs, 'A_FINALIZED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0136",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "A_FINALIZED-COMPLETE"
        ],
        "support": 0.4212,
        "confidence": 0.4212,
        "description": "If 'W_Nabellen offertes-COMPLETE' occurs, 'A_FINALIZED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0137",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "W_Completeren aanvraag-SCHEDULE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'A_FINALIZED-COMPLETE' occurs, 'W_Completeren aanvraag-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0138",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "W_Nabellen offertes-SCHEDULE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'A_FINALIZED-COMPLETE' occurs, 'W_Nabellen offertes-SCHEDULE' must also occur somewhere in the trace"
    },
    {
        "id": "c0139",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "A_PREACCEPTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "If 'O_SENT-COMPLETE' occurs, 'A_PREACCEPTED-COMPLETE' must also occur somewhere in the trace"
    },
    {
        "id": "c0140",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-COMPLETE",
            "W_Completeren aanvraag-START"
        ],
        "support": 0.5967,
        "confidence": 0.5966,
        "description": "If 'W_Completeren aanvraag-COMPLETE' occurs, 'W_Completeren aanvraag-START' must also occur somewhere in the trace"
    },
    {
        "id": "c0302",
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
        "id": "c0303",
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
        "id": "c0343",
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
        "id": "c0344",
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
        "id": "c0356",
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
        "id": "c0410",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'O_CREATED-COMPLETE' and 'O_SENT-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0411",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "A_FINALIZED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'O_CREATED-COMPLETE' and 'A_FINALIZED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0412",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "W_Nabellen offertes-START"
        ],
        "support": 0.4214,
        "confidence": 0.4207,
        "description": "'O_CREATED-COMPLETE' and 'W_Nabellen offertes-START' must both occur, or neither occurs"
    },
    {
        "id": "c0413",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "W_Nabellen offertes-SCHEDULE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'O_CREATED-COMPLETE' and 'W_Nabellen offertes-SCHEDULE' must both occur, or neither occurs"
    },
    {
        "id": "c0414",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "A_ACCEPTED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4214,
        "description": "'O_CREATED-COMPLETE' and 'A_ACCEPTED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0415",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "W_Nabellen offertes-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4212,
        "description": "'O_CREATED-COMPLETE' and 'W_Nabellen offertes-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0416",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "O_SELECTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'O_CREATED-COMPLETE' and 'O_SELECTED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0417",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-START",
            "W_Afhandelen leads-COMPLETE"
        ],
        "support": 0.3551,
        "confidence": 0.3551,
        "description": "'W_Afhandelen leads-START' and 'W_Afhandelen leads-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0418",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-START",
            "W_Afhandelen leads-SCHEDULE"
        ],
        "support": 0.3551,
        "confidence": 0.3551,
        "description": "'W_Afhandelen leads-START' and 'W_Afhandelen leads-SCHEDULE' must both occur, or neither occurs"
    },
    {
        "id": "c0419",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_PARTLYSUBMITTED-COMPLETE",
            "A_SUBMITTED-COMPLETE"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "'A_PARTLYSUBMITTED-COMPLETE' and 'A_SUBMITTED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0420",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "O_CREATED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'O_SENT-COMPLETE' and 'O_CREATED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0421",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "A_FINALIZED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'O_SENT-COMPLETE' and 'A_FINALIZED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0422",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "W_Nabellen offertes-START"
        ],
        "support": 0.4214,
        "confidence": 0.4207,
        "description": "'O_SENT-COMPLETE' and 'W_Nabellen offertes-START' must both occur, or neither occurs"
    },
    {
        "id": "c0423",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "W_Nabellen offertes-SCHEDULE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'O_SENT-COMPLETE' and 'W_Nabellen offertes-SCHEDULE' must both occur, or neither occurs"
    },
    {
        "id": "c0424",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "A_ACCEPTED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4214,
        "description": "'O_SENT-COMPLETE' and 'A_ACCEPTED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0425",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "W_Nabellen offertes-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4212,
        "description": "'O_SENT-COMPLETE' and 'W_Nabellen offertes-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0426",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "O_SELECTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'O_SENT-COMPLETE' and 'O_SELECTED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0427",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "O_CREATED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'A_FINALIZED-COMPLETE' and 'O_CREATED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0428",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'A_FINALIZED-COMPLETE' and 'O_SENT-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0429",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "W_Nabellen offertes-START"
        ],
        "support": 0.4214,
        "confidence": 0.4207,
        "description": "'A_FINALIZED-COMPLETE' and 'W_Nabellen offertes-START' must both occur, or neither occurs"
    },
    {
        "id": "c0430",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "W_Nabellen offertes-SCHEDULE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'A_FINALIZED-COMPLETE' and 'W_Nabellen offertes-SCHEDULE' must both occur, or neither occurs"
    },
    {
        "id": "c0431",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "A_ACCEPTED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4214,
        "description": "'A_FINALIZED-COMPLETE' and 'A_ACCEPTED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0432",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "W_Nabellen offertes-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4212,
        "description": "'A_FINALIZED-COMPLETE' and 'W_Nabellen offertes-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0433",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "O_SELECTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'A_FINALIZED-COMPLETE' and 'O_SELECTED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0434",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "O_CREATED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4207,
        "description": "'W_Nabellen offertes-START' and 'O_CREATED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0435",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4207,
        "description": "'W_Nabellen offertes-START' and 'O_SENT-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0436",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "A_FINALIZED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4207,
        "description": "'W_Nabellen offertes-START' and 'A_FINALIZED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0437",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "W_Nabellen offertes-SCHEDULE"
        ],
        "support": 0.4214,
        "confidence": 0.4207,
        "description": "'W_Nabellen offertes-START' and 'W_Nabellen offertes-SCHEDULE' must both occur, or neither occurs"
    },
    {
        "id": "c0438",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "A_ACCEPTED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4207,
        "description": "'W_Nabellen offertes-START' and 'A_ACCEPTED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0439",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "W_Nabellen offertes-COMPLETE"
        ],
        "support": 0.4212,
        "confidence": 0.4207,
        "description": "'W_Nabellen offertes-START' and 'W_Nabellen offertes-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0440",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "O_SELECTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4207,
        "description": "'W_Nabellen offertes-START' and 'O_SELECTED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0441",
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
        "id": "c0442",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "O_CREATED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'W_Nabellen offertes-SCHEDULE' and 'O_CREATED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0443",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'W_Nabellen offertes-SCHEDULE' and 'O_SENT-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0444",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "A_FINALIZED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'W_Nabellen offertes-SCHEDULE' and 'A_FINALIZED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0445",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "W_Nabellen offertes-START"
        ],
        "support": 0.4214,
        "confidence": 0.4207,
        "description": "'W_Nabellen offertes-SCHEDULE' and 'W_Nabellen offertes-START' must both occur, or neither occurs"
    },
    {
        "id": "c0446",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "A_ACCEPTED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4214,
        "description": "'W_Nabellen offertes-SCHEDULE' and 'A_ACCEPTED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0447",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "W_Nabellen offertes-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4212,
        "description": "'W_Nabellen offertes-SCHEDULE' and 'W_Nabellen offertes-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0448",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "O_SELECTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'W_Nabellen offertes-SCHEDULE' and 'O_SELECTED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0449",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-SCHEDULE",
            "A_PREACCEPTED-COMPLETE"
        ],
        "support": 0.5967,
        "confidence": 0.5967,
        "description": "'W_Completeren aanvraag-SCHEDULE' and 'A_PREACCEPTED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0450",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-SCHEDULE",
            "W_Completeren aanvraag-START"
        ],
        "support": 0.5967,
        "confidence": 0.5966,
        "description": "'W_Completeren aanvraag-SCHEDULE' and 'W_Completeren aanvraag-START' must both occur, or neither occurs"
    },
    {
        "id": "c0451",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-SCHEDULE",
            "W_Completeren aanvraag-COMPLETE"
        ],
        "support": 0.5967,
        "confidence": 0.5967,
        "description": "'W_Completeren aanvraag-SCHEDULE' and 'W_Completeren aanvraag-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0452",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-COMPLETE",
            "W_Afhandelen leads-START"
        ],
        "support": 0.3551,
        "confidence": 0.3551,
        "description": "'W_Afhandelen leads-COMPLETE' and 'W_Afhandelen leads-START' must both occur, or neither occurs"
    },
    {
        "id": "c0453",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-COMPLETE",
            "W_Afhandelen leads-SCHEDULE"
        ],
        "support": 0.3551,
        "confidence": 0.3551,
        "description": "'W_Afhandelen leads-COMPLETE' and 'W_Afhandelen leads-SCHEDULE' must both occur, or neither occurs"
    },
    {
        "id": "c0454",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_PREACCEPTED-COMPLETE",
            "W_Completeren aanvraag-SCHEDULE"
        ],
        "support": 0.5967,
        "confidence": 0.5967,
        "description": "'A_PREACCEPTED-COMPLETE' and 'W_Completeren aanvraag-SCHEDULE' must both occur, or neither occurs"
    },
    {
        "id": "c0455",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_PREACCEPTED-COMPLETE",
            "W_Completeren aanvraag-START"
        ],
        "support": 0.5967,
        "confidence": 0.5966,
        "description": "'A_PREACCEPTED-COMPLETE' and 'W_Completeren aanvraag-START' must both occur, or neither occurs"
    },
    {
        "id": "c0456",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_PREACCEPTED-COMPLETE",
            "W_Completeren aanvraag-COMPLETE"
        ],
        "support": 0.5967,
        "confidence": 0.5967,
        "description": "'A_PREACCEPTED-COMPLETE' and 'W_Completeren aanvraag-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0457",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-START",
            "W_Completeren aanvraag-SCHEDULE"
        ],
        "support": 0.5967,
        "confidence": 0.5966,
        "description": "'W_Completeren aanvraag-START' and 'W_Completeren aanvraag-SCHEDULE' must both occur, or neither occurs"
    },
    {
        "id": "c0458",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-START",
            "A_PREACCEPTED-COMPLETE"
        ],
        "support": 0.5967,
        "confidence": 0.5966,
        "description": "'W_Completeren aanvraag-START' and 'A_PREACCEPTED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0459",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-START",
            "W_Completeren aanvraag-COMPLETE"
        ],
        "support": 0.5967,
        "confidence": 0.5966,
        "description": "'W_Completeren aanvraag-START' and 'W_Completeren aanvraag-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0460",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "O_CREATED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4214,
        "description": "'A_ACCEPTED-COMPLETE' and 'O_CREATED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0461",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4214,
        "description": "'A_ACCEPTED-COMPLETE' and 'O_SENT-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0462",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "A_FINALIZED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4214,
        "description": "'A_ACCEPTED-COMPLETE' and 'A_FINALIZED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0463",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "W_Nabellen offertes-START"
        ],
        "support": 0.4298,
        "confidence": 0.4207,
        "description": "'A_ACCEPTED-COMPLETE' and 'W_Nabellen offertes-START' must both occur, or neither occurs"
    },
    {
        "id": "c0464",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "W_Nabellen offertes-SCHEDULE"
        ],
        "support": 0.4298,
        "confidence": 0.4214,
        "description": "'A_ACCEPTED-COMPLETE' and 'W_Nabellen offertes-SCHEDULE' must both occur, or neither occurs"
    },
    {
        "id": "c0465",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "W_Nabellen offertes-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4212,
        "description": "'A_ACCEPTED-COMPLETE' and 'W_Nabellen offertes-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0466",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "O_SELECTED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4214,
        "description": "'A_ACCEPTED-COMPLETE' and 'O_SELECTED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0467",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-SCHEDULE",
            "W_Afhandelen leads-START"
        ],
        "support": 0.3551,
        "confidence": 0.3551,
        "description": "'W_Afhandelen leads-SCHEDULE' and 'W_Afhandelen leads-START' must both occur, or neither occurs"
    },
    {
        "id": "c0468",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-SCHEDULE",
            "W_Afhandelen leads-COMPLETE"
        ],
        "support": 0.3551,
        "confidence": 0.3551,
        "description": "'W_Afhandelen leads-SCHEDULE' and 'W_Afhandelen leads-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0469",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-COMPLETE",
            "W_Completeren aanvraag-SCHEDULE"
        ],
        "support": 0.5967,
        "confidence": 0.5967,
        "description": "'W_Completeren aanvraag-COMPLETE' and 'W_Completeren aanvraag-SCHEDULE' must both occur, or neither occurs"
    },
    {
        "id": "c0470",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-COMPLETE",
            "A_PREACCEPTED-COMPLETE"
        ],
        "support": 0.5967,
        "confidence": 0.5967,
        "description": "'W_Completeren aanvraag-COMPLETE' and 'A_PREACCEPTED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0471",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-COMPLETE",
            "W_Completeren aanvraag-START"
        ],
        "support": 0.5967,
        "confidence": 0.5966,
        "description": "'W_Completeren aanvraag-COMPLETE' and 'W_Completeren aanvraag-START' must both occur, or neither occurs"
    },
    {
        "id": "c0472",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "O_CREATED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4212,
        "description": "'W_Nabellen offertes-COMPLETE' and 'O_CREATED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0473",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4212,
        "description": "'W_Nabellen offertes-COMPLETE' and 'O_SENT-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0474",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "A_FINALIZED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4212,
        "description": "'W_Nabellen offertes-COMPLETE' and 'A_FINALIZED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0475",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "W_Nabellen offertes-START"
        ],
        "support": 0.4212,
        "confidence": 0.4207,
        "description": "'W_Nabellen offertes-COMPLETE' and 'W_Nabellen offertes-START' must both occur, or neither occurs"
    },
    {
        "id": "c0476",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "W_Nabellen offertes-SCHEDULE"
        ],
        "support": 0.4214,
        "confidence": 0.4212,
        "description": "'W_Nabellen offertes-COMPLETE' and 'W_Nabellen offertes-SCHEDULE' must both occur, or neither occurs"
    },
    {
        "id": "c0477",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "A_ACCEPTED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4212,
        "description": "'W_Nabellen offertes-COMPLETE' and 'A_ACCEPTED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0478",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "O_SELECTED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4212,
        "description": "'W_Nabellen offertes-COMPLETE' and 'O_SELECTED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0479",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "O_CREATED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'O_SELECTED-COMPLETE' and 'O_CREATED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0480",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'O_SELECTED-COMPLETE' and 'O_SENT-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0481",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "A_FINALIZED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'O_SELECTED-COMPLETE' and 'A_FINALIZED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0482",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "W_Nabellen offertes-START"
        ],
        "support": 0.4214,
        "confidence": 0.4207,
        "description": "'O_SELECTED-COMPLETE' and 'W_Nabellen offertes-START' must both occur, or neither occurs"
    },
    {
        "id": "c0483",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "W_Nabellen offertes-SCHEDULE"
        ],
        "support": 0.4214,
        "confidence": 0.4214,
        "description": "'O_SELECTED-COMPLETE' and 'W_Nabellen offertes-SCHEDULE' must both occur, or neither occurs"
    },
    {
        "id": "c0484",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "A_ACCEPTED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.4214,
        "description": "'O_SELECTED-COMPLETE' and 'A_ACCEPTED-COMPLETE' must both occur, or neither occurs"
    },
    {
        "id": "c0485",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "W_Nabellen offertes-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 0.4212,
        "description": "'O_SELECTED-COMPLETE' and 'W_Nabellen offertes-COMPLETE' must both occur, or neither occurs"
    }
]

# Ordered list of all unique activities referenced by the constraints above
UNIQUE_ACTIVITIES: list[str] = ['A_ACCEPTED-COMPLETE', 'O_SELECTED-COMPLETE', 'O_SENT-COMPLETE', 'W_Nabellen offertes-COMPLETE', 'W_Nabellen offertes-START', 'W_Completeren aanvraag-SCHEDULE', 'A_FINALIZED-COMPLETE', 'W_Nabellen offertes-SCHEDULE', 'O_CREATED-COMPLETE', 'W_Afhandelen leads-SCHEDULE', 'A_SUBMITTED-COMPLETE', 'W_Completeren aanvraag-COMPLETE', 'A_PARTLYSUBMITTED-COMPLETE', 'W_Completeren aanvraag-START', 'A_PREACCEPTED-COMPLETE', 'W_Afhandelen leads-START', 'A_DECLINED-COMPLETE', 'W_Afhandelen leads-COMPLETE']


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
