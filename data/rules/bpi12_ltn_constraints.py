"""
Auto-generated LTN constraints from DECLARE rules.

Dataset:   bpi12
Mode:      discriminability filter  (csv=BPI12_student_model_val_declare_discriminability.csv, min_net=1)
Total:     82 constraints  (altprecedence=14, altresponse=12, altsuccession=9, chainprecedence=1, chainresponse=2, chainsuccession=2, noncoexistence=40, precedence=2)

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
        "id": "c0022",
        "template": "precedence",
        "arity": "binary",
        "activities": [
            "A_PREACCEPTED-COMPLETE",
            "W_Completeren aanvraag-START"
        ],
        "support": 0.5966,
        "confidence": 1.0,
        "category": "ordering",
        "description": "'W_Completeren aanvraag-START' can only occur if 'A_PREACCEPTED-COMPLETE' has occurred before it"
    },
    {
        "id": "c0108",
        "template": "precedence",
        "arity": "binary",
        "activities": [
            "W_Nabellen incomplete dossiers-SCHEDULE",
            "W_Nabellen incomplete dossiers-START"
        ],
        "support": 0.1479,
        "confidence": 1.0,
        "category": "ordering",
        "description": "'W_Nabellen incomplete dossiers-START' can only occur if 'W_Nabellen incomplete dossiers-SCHEDULE' has occurred before it"
    },
    {
        "id": "c0302",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'O_SENT-COMPLETE' must be preceded by 'O_SELECTED-COMPLETE' with no other 'O_SENT-COMPLETE' in between"
    },
    {
        "id": "c0320",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "W_Nabellen offertes-SCHEDULE"
        ],
        "support": 0.4214,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'W_Nabellen offertes-SCHEDULE' must be preceded by 'O_SELECTED-COMPLETE' with no other 'W_Nabellen offertes-SCHEDULE' in between"
    },
    {
        "id": "c0327",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "O_CANCELLED-COMPLETE"
        ],
        "support": 0.2235,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'O_CANCELLED-COMPLETE' must be preceded by 'O_SENT-COMPLETE' with no other 'O_CANCELLED-COMPLETE' in between"
    },
    {
        "id": "c0330",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "W_Nabellen incomplete dossiers-START",
            "W_Nabellen incomplete dossiers-COMPLETE"
        ],
        "support": 0.1479,
        "confidence": 0.9944,
        "category": "ordering",
        "description": "Each 'W_Nabellen incomplete dossiers-COMPLETE' must be preceded by 'W_Nabellen incomplete dossiers-START' with no other 'W_Nabellen incomplete dossiers-COMPLETE' in between"
    },
    {
        "id": "c0334",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "O_CREATED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'O_CREATED-COMPLETE' must be preceded by 'O_SELECTED-COMPLETE' with no other 'O_CREATED-COMPLETE' in between"
    },
    {
        "id": "c0342",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-START",
            "W_Completeren aanvraag-COMPLETE"
        ],
        "support": 0.5967,
        "confidence": 0.9532,
        "category": "ordering",
        "description": "Each 'W_Completeren aanvraag-COMPLETE' must be preceded by 'W_Completeren aanvraag-START' with no other 'W_Completeren aanvraag-COMPLETE' in between"
    },
    {
        "id": "c0346",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "O_CANCELLED-COMPLETE"
        ],
        "support": 0.2235,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'O_CANCELLED-COMPLETE' must be preceded by 'O_CREATED-COMPLETE' with no other 'O_CANCELLED-COMPLETE' in between"
    },
    {
        "id": "c0349",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'O_SENT-COMPLETE' must be preceded by 'O_CREATED-COMPLETE' with no other 'O_SENT-COMPLETE' in between"
    },
    {
        "id": "c0351",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "A_PARTLYSUBMITTED-COMPLETE",
            "A_ACCEPTED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'A_ACCEPTED-COMPLETE' must be preceded by 'A_PARTLYSUBMITTED-COMPLETE' with no other 'A_ACCEPTED-COMPLETE' in between"
    },
    {
        "id": "c0374",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-SCHEDULE",
            "A_ACCEPTED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'A_ACCEPTED-COMPLETE' must be preceded by 'W_Completeren aanvraag-SCHEDULE' with no other 'A_ACCEPTED-COMPLETE' in between"
    },
    {
        "id": "c0385",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-START",
            "W_Afhandelen leads-COMPLETE"
        ],
        "support": 0.3551,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'W_Afhandelen leads-COMPLETE' must be preceded by 'W_Afhandelen leads-START' with no other 'W_Afhandelen leads-COMPLETE' in between"
    },
    {
        "id": "c0402",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "A_PREACCEPTED-COMPLETE",
            "A_ACCEPTED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'A_ACCEPTED-COMPLETE' must be preceded by 'A_PREACCEPTED-COMPLETE' with no other 'A_ACCEPTED-COMPLETE' in between"
    },
    {
        "id": "c0403",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "A_SUBMITTED-COMPLETE",
            "A_ACCEPTED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'A_ACCEPTED-COMPLETE' must be preceded by 'A_SUBMITTED-COMPLETE' with no other 'A_ACCEPTED-COMPLETE' in between"
    },
    {
        "id": "c0410",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "W_Valideren aanvraag-START",
            "W_Valideren aanvraag-COMPLETE"
        ],
        "support": 0.272,
        "confidence": 0.9987,
        "category": "ordering",
        "description": "Each 'W_Valideren aanvraag-COMPLETE' must be preceded by 'W_Valideren aanvraag-START' with no other 'W_Valideren aanvraag-COMPLETE' in between"
    },
    {
        "id": "c0421",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "W_Nabellen offertes-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.98,
        "category": "ordering",
        "description": "Whenever 'A_ACCEPTED-COMPLETE' occurs, 'W_Nabellen offertes-COMPLETE' must follow before 'A_ACCEPTED-COMPLETE' can recur"
    },
    {
        "id": "c0424",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "W_Nabellen offertes-SCHEDULE"
        ],
        "support": 0.4214,
        "confidence": 0.9263,
        "category": "ordering",
        "description": "Whenever 'O_SELECTED-COMPLETE' occurs, 'W_Nabellen offertes-SCHEDULE' must follow before 'O_SELECTED-COMPLETE' can recur"
    },
    {
        "id": "c0428",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "W_Nabellen offertes-START"
        ],
        "support": 0.4298,
        "confidence": 0.9789,
        "category": "ordering",
        "description": "Whenever 'A_ACCEPTED-COMPLETE' occurs, 'W_Nabellen offertes-START' must follow before 'A_ACCEPTED-COMPLETE' can recur"
    },
    {
        "id": "c0432",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "A_FINALIZED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.9806,
        "category": "ordering",
        "description": "Whenever 'A_ACCEPTED-COMPLETE' occurs, 'A_FINALIZED-COMPLETE' must follow before 'A_ACCEPTED-COMPLETE' can recur"
    },
    {
        "id": "c0446",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "W_Nabellen offertes-SCHEDULE"
        ],
        "support": 0.4298,
        "confidence": 0.9806,
        "category": "ordering",
        "description": "Whenever 'A_ACCEPTED-COMPLETE' occurs, 'W_Nabellen offertes-SCHEDULE' must follow before 'A_ACCEPTED-COMPLETE' can recur"
    },
    {
        "id": "c0447",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "O_SELECTED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.9806,
        "category": "ordering",
        "description": "Whenever 'A_ACCEPTED-COMPLETE' occurs, 'O_SELECTED-COMPLETE' must follow before 'A_ACCEPTED-COMPLETE' can recur"
    },
    {
        "id": "c0450",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "W_Nabellen offertes-START"
        ],
        "support": 0.4214,
        "confidence": 0.9235,
        "category": "ordering",
        "description": "Whenever 'O_SELECTED-COMPLETE' occurs, 'W_Nabellen offertes-START' must follow before 'O_SELECTED-COMPLETE' can recur"
    },
    {
        "id": "c0456",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "O_CREATED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.9806,
        "category": "ordering",
        "description": "Whenever 'A_ACCEPTED-COMPLETE' occurs, 'O_CREATED-COMPLETE' must follow before 'A_ACCEPTED-COMPLETE' can recur"
    },
    {
        "id": "c0459",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "O_CREATED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Whenever 'O_SELECTED-COMPLETE' occurs, 'O_CREATED-COMPLETE' must follow before 'O_SELECTED-COMPLETE' can recur"
    },
    {
        "id": "c0463",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.9806,
        "category": "ordering",
        "description": "Whenever 'A_ACCEPTED-COMPLETE' occurs, 'O_SENT-COMPLETE' must follow before 'A_ACCEPTED-COMPLETE' can recur"
    },
    {
        "id": "c0465",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "W_Nabellen offertes-START"
        ],
        "support": 0.4214,
        "confidence": 0.9972,
        "category": "ordering",
        "description": "Whenever 'W_Nabellen offertes-SCHEDULE' occurs, 'W_Nabellen offertes-START' must follow before 'W_Nabellen offertes-SCHEDULE' can recur"
    },
    {
        "id": "c0468",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Whenever 'O_SELECTED-COMPLETE' occurs, 'O_SENT-COMPLETE' must follow before 'O_SELECTED-COMPLETE' can recur"
    },
    {
        "id": "c0476",
        "template": "chainprecedence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 1.0,
        "category": "immediate",
        "description": "'O_SENT-COMPLETE' can only occur if 'O_CREATED-COMPLETE' immediately preceded it"
    },
    {
        "id": "c0478",
        "template": "chainresponse",
        "arity": "binary",
        "activities": [
            "A_PREACCEPTED-COMPLETE",
            "W_Completeren aanvraag-SCHEDULE"
        ],
        "support": 0.5967,
        "confidence": 1.0,
        "category": "immediate",
        "description": "Whenever 'A_PREACCEPTED-COMPLETE' occurs, 'W_Completeren aanvraag-SCHEDULE' must immediately follow"
    },
    {
        "id": "c0479",
        "template": "chainresponse",
        "arity": "binary",
        "activities": [
            "O_SENT_BACK-COMPLETE",
            "W_Valideren aanvraag-SCHEDULE"
        ],
        "support": 0.2758,
        "confidence": 1.0,
        "category": "immediate",
        "description": "Whenever 'O_SENT_BACK-COMPLETE' occurs, 'W_Valideren aanvraag-SCHEDULE' must immediately follow"
    },
    {
        "id": "c0491",
        "template": "altsuccession",
        "arity": "binary",
        "activities": [
            "W_Valideren aanvraag-START",
            "W_Valideren aanvraag-COMPLETE"
        ],
        "support": 0.272,
        "confidence": 0.9987,
        "category": "ordering",
        "description": "'W_Valideren aanvraag-START' and 'W_Valideren aanvraag-COMPLETE' alternate: each 'W_Valideren aanvraag-START' triggers exactly one 'W_Valideren aanvraag-COMPLETE' and vice versa"
    },
    {
        "id": "c0492",
        "template": "altsuccession",
        "arity": "binary",
        "activities": [
            "W_Afhandelen leads-START",
            "W_Afhandelen leads-COMPLETE"
        ],
        "support": 0.3551,
        "confidence": 1.0,
        "category": "ordering",
        "description": "'W_Afhandelen leads-START' and 'W_Afhandelen leads-COMPLETE' alternate: each 'W_Afhandelen leads-START' triggers exactly one 'W_Afhandelen leads-COMPLETE' and vice versa"
    },
    {
        "id": "c0495",
        "template": "altsuccession",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "A_FINALIZED-COMPLETE"
        ],
        "support": 0.4298,
        "confidence": 0.9806,
        "category": "ordering",
        "description": "'A_ACCEPTED-COMPLETE' and 'A_FINALIZED-COMPLETE' alternate: each 'A_ACCEPTED-COMPLETE' triggers exactly one 'A_FINALIZED-COMPLETE' and vice versa"
    },
    {
        "id": "c0496",
        "template": "altsuccession",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "W_Nabellen offertes-SCHEDULE"
        ],
        "support": 0.4214,
        "confidence": 0.9263,
        "category": "ordering",
        "description": "'O_SELECTED-COMPLETE' and 'W_Nabellen offertes-SCHEDULE' alternate: each 'O_SELECTED-COMPLETE' triggers exactly one 'W_Nabellen offertes-SCHEDULE' and vice versa"
    },
    {
        "id": "c0497",
        "template": "altsuccession",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 1.0,
        "category": "ordering",
        "description": "'O_SELECTED-COMPLETE' and 'O_SENT-COMPLETE' alternate: each 'O_SELECTED-COMPLETE' triggers exactly one 'O_SENT-COMPLETE' and vice versa"
    },
    {
        "id": "c0498",
        "template": "altsuccession",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "O_CREATED-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 1.0,
        "category": "ordering",
        "description": "'O_SELECTED-COMPLETE' and 'O_CREATED-COMPLETE' alternate: each 'O_SELECTED-COMPLETE' triggers exactly one 'O_CREATED-COMPLETE' and vice versa"
    },
    {
        "id": "c0499",
        "template": "altsuccession",
        "arity": "binary",
        "activities": [
            "W_Nabellen incomplete dossiers-START",
            "W_Nabellen incomplete dossiers-COMPLETE"
        ],
        "support": 0.1479,
        "confidence": 0.9944,
        "category": "ordering",
        "description": "'W_Nabellen incomplete dossiers-START' and 'W_Nabellen incomplete dossiers-COMPLETE' alternate: each 'W_Nabellen incomplete dossiers-START' triggers exactly one 'W_Nabellen incomplete dossiers-COMPLETE' and vice versa"
    },
    {
        "id": "c0500",
        "template": "altsuccession",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-START",
            "W_Completeren aanvraag-COMPLETE"
        ],
        "support": 0.5967,
        "confidence": 0.9532,
        "category": "ordering",
        "description": "'W_Completeren aanvraag-START' and 'W_Completeren aanvraag-COMPLETE' alternate: each 'W_Completeren aanvraag-START' triggers exactly one 'W_Completeren aanvraag-COMPLETE' and vice versa"
    },
    {
        "id": "c0503",
        "template": "altsuccession",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 1.0,
        "category": "ordering",
        "description": "'O_CREATED-COMPLETE' and 'O_SENT-COMPLETE' alternate: each 'O_CREATED-COMPLETE' triggers exactly one 'O_SENT-COMPLETE' and vice versa"
    },
    {
        "id": "c0505",
        "template": "chainsuccession",
        "arity": "binary",
        "activities": [
            "A_PREACCEPTED-COMPLETE",
            "W_Completeren aanvraag-SCHEDULE"
        ],
        "support": 0.5967,
        "confidence": 0.9994,
        "category": "immediate",
        "description": "'A_PREACCEPTED-COMPLETE' and 'W_Completeren aanvraag-SCHEDULE' must always occur as consecutive pairs"
    },
    {
        "id": "c0507",
        "template": "chainsuccession",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4214,
        "confidence": 1.0,
        "category": "immediate",
        "description": "'O_CREATED-COMPLETE' and 'O_SENT-COMPLETE' must always occur as consecutive pairs"
    },
    {
        "id": "c0511",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-COMPLETE",
            "W_Beoordelen fraude-COMPLETE"
        ],
        "support": 0.4271,
        "confidence": 0.9925,
        "category": "cross_path",
        "description": "'W_Nabellen offertes-COMPLETE' and 'W_Beoordelen fraude-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0515",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Valideren aanvraag-START",
            "W_Beoordelen fraude-COMPLETE"
        ],
        "support": 0.2776,
        "confidence": 0.9884,
        "category": "cross_path",
        "description": "'W_Valideren aanvraag-START' and 'W_Beoordelen fraude-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0520",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Valideren aanvraag-COMPLETE",
            "W_Beoordelen fraude-COMPLETE"
        ],
        "support": 0.2778,
        "confidence": 0.9884,
        "category": "cross_path",
        "description": "'W_Valideren aanvraag-COMPLETE' and 'W_Beoordelen fraude-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0525",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "O_CANCELLED-COMPLETE",
            "W_Beoordelen fraude-COMPLETE"
        ],
        "support": 0.2315,
        "confidence": 0.9954,
        "category": "cross_path",
        "description": "'O_CANCELLED-COMPLETE' and 'W_Beoordelen fraude-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0545",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-SCHEDULE",
            "W_Beoordelen fraude-COMPLETE"
        ],
        "support": 0.4273,
        "confidence": 0.9925,
        "category": "cross_path",
        "description": "'W_Nabellen offertes-SCHEDULE' and 'W_Beoordelen fraude-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0557",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "A_PREACCEPTED-COMPLETE",
            "W_Beoordelen fraude-COMPLETE"
        ],
        "support": 0.6021,
        "confidence": 0.9939,
        "category": "cross_path",
        "description": "'A_PREACCEPTED-COMPLETE' and 'W_Beoordelen fraude-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0561",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "A_ACCEPTED-COMPLETE",
            "W_Beoordelen fraude-COMPLETE"
        ],
        "support": 0.4355,
        "confidence": 0.9923,
        "category": "cross_path",
        "description": "'A_ACCEPTED-COMPLETE' and 'W_Beoordelen fraude-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0572",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "O_SENT_BACK-COMPLETE",
            "W_Beoordelen fraude-COMPLETE"
        ],
        "support": 0.2816,
        "confidence": 0.9886,
        "category": "cross_path",
        "description": "'O_SENT_BACK-COMPLETE' and 'W_Beoordelen fraude-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0588",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-START",
            "O_SELECTED-COMPLETE"
        ],
        "support": 0.4273,
        "confidence": 0.9925,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-START' and 'O_SELECTED-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0600",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-START",
            "O_ACCEPTED-COMPLETE"
        ],
        "support": 0.1962,
        "confidence": 0.986,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-START' and 'O_ACCEPTED-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0605",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "W_Beoordelen fraude-START"
        ],
        "support": 0.4273,
        "confidence": 0.9925,
        "category": "cross_path",
        "description": "'O_SELECTED-COMPLETE' and 'W_Beoordelen fraude-START' cannot both occur in the same trace"
    },
    {
        "id": "c0607",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "W_Beoordelen fraude-SCHEDULE"
        ],
        "support": 0.4273,
        "confidence": 0.9925,
        "category": "cross_path",
        "description": "'O_SELECTED-COMPLETE' and 'W_Beoordelen fraude-SCHEDULE' cannot both occur in the same trace"
    },
    {
        "id": "c0608",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "O_SELECTED-COMPLETE",
            "W_Beoordelen fraude-COMPLETE"
        ],
        "support": 0.4273,
        "confidence": 0.9925,
        "category": "cross_path",
        "description": "'O_SELECTED-COMPLETE' and 'W_Beoordelen fraude-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0682",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-SCHEDULE",
            "O_SELECTED-COMPLETE"
        ],
        "support": 0.4273,
        "confidence": 0.9925,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-SCHEDULE' and 'O_SELECTED-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0694",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-SCHEDULE",
            "O_ACCEPTED-COMPLETE"
        ],
        "support": 0.1962,
        "confidence": 0.986,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-SCHEDULE' and 'O_ACCEPTED-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0702",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Completeren aanvraag-SCHEDULE",
            "W_Beoordelen fraude-COMPLETE"
        ],
        "support": 0.6021,
        "confidence": 0.9939,
        "category": "cross_path",
        "description": "'W_Completeren aanvraag-SCHEDULE' and 'W_Beoordelen fraude-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0703",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-COMPLETE",
            "W_Nabellen offertes-COMPLETE"
        ],
        "support": 0.4271,
        "confidence": 0.9925,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-COMPLETE' and 'W_Nabellen offertes-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0704",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-COMPLETE",
            "W_Valideren aanvraag-START"
        ],
        "support": 0.2776,
        "confidence": 0.9884,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-COMPLETE' and 'W_Valideren aanvraag-START' cannot both occur in the same trace"
    },
    {
        "id": "c0705",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-COMPLETE",
            "W_Valideren aanvraag-COMPLETE"
        ],
        "support": 0.2778,
        "confidence": 0.9884,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-COMPLETE' and 'W_Valideren aanvraag-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0706",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-COMPLETE",
            "O_CANCELLED-COMPLETE"
        ],
        "support": 0.2315,
        "confidence": 0.9954,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-COMPLETE' and 'O_CANCELLED-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0710",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-COMPLETE",
            "W_Nabellen offertes-SCHEDULE"
        ],
        "support": 0.4273,
        "confidence": 0.9925,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-COMPLETE' and 'W_Nabellen offertes-SCHEDULE' cannot both occur in the same trace"
    },
    {
        "id": "c0713",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-COMPLETE",
            "A_PREACCEPTED-COMPLETE"
        ],
        "support": 0.6021,
        "confidence": 0.9939,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-COMPLETE' and 'A_PREACCEPTED-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0714",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-COMPLETE",
            "A_ACCEPTED-COMPLETE"
        ],
        "support": 0.4355,
        "confidence": 0.9923,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-COMPLETE' and 'A_ACCEPTED-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0716",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-COMPLETE",
            "O_SENT_BACK-COMPLETE"
        ],
        "support": 0.2816,
        "confidence": 0.9886,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-COMPLETE' and 'O_SENT_BACK-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0717",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-COMPLETE",
            "O_SELECTED-COMPLETE"
        ],
        "support": 0.4273,
        "confidence": 0.9925,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-COMPLETE' and 'O_SELECTED-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0722",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-COMPLETE",
            "W_Completeren aanvraag-SCHEDULE"
        ],
        "support": 0.6021,
        "confidence": 0.9939,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-COMPLETE' and 'W_Completeren aanvraag-SCHEDULE' cannot both occur in the same trace"
    },
    {
        "id": "c0724",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-COMPLETE",
            "W_Nabellen offertes-START"
        ],
        "support": 0.4266,
        "confidence": 0.9924,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-COMPLETE' and 'W_Nabellen offertes-START' cannot both occur in the same trace"
    },
    {
        "id": "c0728",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-COMPLETE",
            "O_SENT-COMPLETE"
        ],
        "support": 0.4273,
        "confidence": 0.9925,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-COMPLETE' and 'O_SENT-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0729",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-COMPLETE",
            "O_ACCEPTED-COMPLETE"
        ],
        "support": 0.1962,
        "confidence": 0.986,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-COMPLETE' and 'O_ACCEPTED-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0730",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-COMPLETE",
            "O_CREATED-COMPLETE"
        ],
        "support": 0.4273,
        "confidence": 0.9925,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-COMPLETE' and 'O_CREATED-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0731",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-COMPLETE",
            "A_FINALIZED-COMPLETE"
        ],
        "support": 0.4273,
        "confidence": 0.9925,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-COMPLETE' and 'A_FINALIZED-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0733",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Beoordelen fraude-COMPLETE",
            "W_Valideren aanvraag-SCHEDULE"
        ],
        "support": 0.2816,
        "confidence": 0.9886,
        "category": "cross_path",
        "description": "'W_Beoordelen fraude-COMPLETE' and 'W_Valideren aanvraag-SCHEDULE' cannot both occur in the same trace"
    },
    {
        "id": "c0741",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Nabellen offertes-START",
            "W_Beoordelen fraude-COMPLETE"
        ],
        "support": 0.4266,
        "confidence": 0.9924,
        "category": "cross_path",
        "description": "'W_Nabellen offertes-START' and 'W_Beoordelen fraude-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0772",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "O_SENT-COMPLETE",
            "W_Beoordelen fraude-COMPLETE"
        ],
        "support": 0.4273,
        "confidence": 0.9925,
        "category": "cross_path",
        "description": "'O_SENT-COMPLETE' and 'W_Beoordelen fraude-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0773",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "O_ACCEPTED-COMPLETE",
            "W_Beoordelen fraude-START"
        ],
        "support": 0.1962,
        "confidence": 0.986,
        "category": "cross_path",
        "description": "'O_ACCEPTED-COMPLETE' and 'W_Beoordelen fraude-START' cannot both occur in the same trace"
    },
    {
        "id": "c0776",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "O_ACCEPTED-COMPLETE",
            "W_Beoordelen fraude-SCHEDULE"
        ],
        "support": 0.1962,
        "confidence": 0.986,
        "category": "cross_path",
        "description": "'O_ACCEPTED-COMPLETE' and 'W_Beoordelen fraude-SCHEDULE' cannot both occur in the same trace"
    },
    {
        "id": "c0777",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "O_ACCEPTED-COMPLETE",
            "W_Beoordelen fraude-COMPLETE"
        ],
        "support": 0.1962,
        "confidence": 0.986,
        "category": "cross_path",
        "description": "'O_ACCEPTED-COMPLETE' and 'W_Beoordelen fraude-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0783",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "O_CREATED-COMPLETE",
            "W_Beoordelen fraude-COMPLETE"
        ],
        "support": 0.4273,
        "confidence": 0.9925,
        "category": "cross_path",
        "description": "'O_CREATED-COMPLETE' and 'W_Beoordelen fraude-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0787",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "A_FINALIZED-COMPLETE",
            "W_Beoordelen fraude-COMPLETE"
        ],
        "support": 0.4273,
        "confidence": 0.9925,
        "category": "cross_path",
        "description": "'A_FINALIZED-COMPLETE' and 'W_Beoordelen fraude-COMPLETE' cannot both occur in the same trace"
    },
    {
        "id": "c0804",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "W_Valideren aanvraag-SCHEDULE",
            "W_Beoordelen fraude-COMPLETE"
        ],
        "support": 0.2816,
        "confidence": 0.9886,
        "category": "cross_path",
        "description": "'W_Valideren aanvraag-SCHEDULE' and 'W_Beoordelen fraude-COMPLETE' cannot both occur in the same trace"
    }
]

# Ordered list of all unique activities referenced by the constraints above
UNIQUE_ACTIVITIES: list[str] = ['A_PREACCEPTED-COMPLETE', 'W_Completeren aanvraag-START', 'W_Nabellen incomplete dossiers-SCHEDULE', 'W_Nabellen incomplete dossiers-START', 'O_SELECTED-COMPLETE', 'O_SENT-COMPLETE', 'W_Nabellen offertes-SCHEDULE', 'O_CANCELLED-COMPLETE', 'W_Nabellen incomplete dossiers-COMPLETE', 'O_CREATED-COMPLETE', 'W_Completeren aanvraag-COMPLETE', 'A_PARTLYSUBMITTED-COMPLETE', 'A_ACCEPTED-COMPLETE', 'W_Completeren aanvraag-SCHEDULE', 'W_Afhandelen leads-START', 'W_Afhandelen leads-COMPLETE', 'A_SUBMITTED-COMPLETE', 'W_Valideren aanvraag-START', 'W_Valideren aanvraag-COMPLETE', 'W_Nabellen offertes-COMPLETE', 'W_Nabellen offertes-START', 'A_FINALIZED-COMPLETE', 'O_SENT_BACK-COMPLETE', 'W_Valideren aanvraag-SCHEDULE', 'W_Beoordelen fraude-COMPLETE', 'W_Beoordelen fraude-START', 'O_ACCEPTED-COMPLETE', 'W_Beoordelen fraude-SCHEDULE']


# ── Level 3: shared predicates (architecture) ────────────────────────────────
def make_predicates(activity_col_start: int, seq_len: int) -> dict:
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
    Returns a float tensor of shape (batch_size, N_active_constraints).
    1.0 = constraint satisfied by the prefix, 0.0 = violated.
    Constraints whose activities are absent from activity_vocab are skipped.
    """
    s, e = activity_col_start, activity_col_start + seq_len
    act = x[:, s:e].long()
    B, T = act.size()
    device = act.device
    features: list[torch.Tensor] = []

    for c in CONSTRAINTS:
        t    = c["template"]
        acts = c["activities"]

        # ── Unary ──────────────────────────────────────────────────────────
        if t in ("existence", "init", "exactly_one", "absence"):
            code = activity_vocab.get(acts[0])
            if code is None:
                continue
            if t == "existence":
                features.append((act == code).any(dim=1).float())
            elif t == "init":
                features.append((act[:, 0] == code).float())
            elif t == "exactly_one":
                features.append(((act == code).sum(dim=1) == 1).float())
            elif t == "absence":
                features.append((~(act == code).any(dim=1)).float())

        # ── Binary coexistence / exclusion ─────────────────────────────────
        elif t in ("responded_existence", "coexistence", "noncoexistence"):
            code_a = activity_vocab.get(acts[0])
            code_b = activity_vocab.get(acts[1])
            if code_a is None or code_b is None:
                continue
            has_a = (act == code_a).any(dim=1)
            has_b = (act == code_b).any(dim=1)
            if t == "responded_existence":
                features.append((~has_a | has_b).float())
            elif t == "coexistence":
                features.append((has_a == has_b).float())
            elif t == "noncoexistence":
                features.append((~(has_a & has_b)).float())

        # ── Precedence ─────────────────────────────────────────────────────
        elif t == "precedence":
            code_a = activity_vocab.get(acts[0])
            code_b = activity_vocab.get(acts[1])
            if code_a is None or code_b is None:
                continue
            _INF = T
            _idx = torch.arange(T, device=device).unsqueeze(0)
            first_a = torch.where(act == code_a, _idx,
                                  torch.tensor(_INF, device=device)).min(1).values
            first_b = torch.where(act == code_b, _idx,
                                  torch.tensor(_INF, device=device)).min(1).values
            violated = (first_b < _INF) & (first_b < first_a)
            features.append((~violated).float())

        # ── Alternating response ───────────────────────────────────────────
        elif t == "altresponse":
            code_a = activity_vocab.get(acts[0])
            code_b = activity_vocab.get(acts[1])
            if code_a is None or code_b is None:
                continue
            pos_a = (act == code_a)
            pos_b = (act == code_b)
            s = pos_a.long().cumsum(dim=1) - pos_b.long().cumsum(dim=1)
            s_before = torch.cat([torch.zeros(B, 1, dtype=s.dtype, device=device), s[:, :-1]], dim=1)
            violated = (pos_a & (s_before > 0)).any(dim=1)
            features.append((~violated).float())

        # ── Alternating precedence ─────────────────────────────────────────
        elif t == "altprecedence":
            code_a = activity_vocab.get(acts[0])
            code_b = activity_vocab.get(acts[1])
            if code_a is None or code_b is None:
                continue
            pos_a = (act == code_a)
            pos_b = (act == code_b)
            s = pos_a.long().cumsum(dim=1) - pos_b.long().cumsum(dim=1)
            s_before = torch.cat([torch.zeros(B, 1, dtype=s.dtype, device=device), s[:, :-1]], dim=1)
            violated = (pos_b & (s_before <= 0)).any(dim=1)
            features.append((~violated).float())

        # ── Alternating succession (altprecedence AND altresponse) ──────────
        elif t == "altsuccession":
            code_a = activity_vocab.get(acts[0])
            code_b = activity_vocab.get(acts[1])
            if code_a is None or code_b is None:
                continue
            pos_a = (act == code_a)
            pos_b = (act == code_b)
            s = pos_a.long().cumsum(dim=1) - pos_b.long().cumsum(dim=1)
            s_before = torch.cat([torch.zeros(B, 1, dtype=s.dtype, device=device), s[:, :-1]], dim=1)
            vr = (pos_a & (s_before > 0)).any(dim=1)
            vp = (pos_b & (s_before <= 0)).any(dim=1)
            features.append((~(vr | vp)).float())

        # ── Chain constraints (immediate succession) ───────────────────────
        elif t in ("chainresponse", "chainprecedence", "chainsuccession"):
            code_a = activity_vocab.get(acts[0])
            code_b = activity_vocab.get(acts[1])
            if code_a is None or code_b is None:
                continue
            if t == "chainresponse":
                # violated if A appears but next position is not B
                is_a    = act[:, :-1] == code_a
                next_nb = act[:, 1:]  != code_b
                violated = (is_a & next_nb).any(dim=1)
            elif t == "chainprecedence":
                # violated if B appears without A immediately before it
                is_b    = act[:, 1:]  == code_b
                prev_na = act[:, :-1] != code_a
                violated = (is_b & prev_na).any(dim=1) | (act[:, 0] == code_b)
            else:  # chainsuccession
                is_a_r  = act[:, :-1] == code_a
                next_nb = act[:, 1:]  != code_b
                viol_r  = (is_a_r & next_nb).any(dim=1)
                is_b_p  = act[:, 1:]  == code_b
                prev_na = act[:, :-1] != code_a
                viol_p  = (is_b_p & prev_na).any(dim=1) | (act[:, 0] == code_b)
                violated = viol_r | viol_p
            features.append((~violated).float())

    if not features:
        return torch.zeros(x.size(0), 0, device=x.device)
    return torch.stack(features, dim=1)


# ── Level 2: LTN loss formulas (existence-based templates only) ──────────────
def build_level2_formulas(
    x_var, predicates, constants,
    Forall, Implies, Not, And, Or,
    activity_col_start: int = 0,
    seq_len: int = 40,
) -> list:
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
            if const_a is None: continue
            formulas.append(Forall(x_var, HasAct(x_var, const_a)))
        elif t == "exactly_one":
            const_a = constants.get(acts[0])
            if const_a is None: continue
            formulas.append(Forall(x_var, ExactlyOnce(x_var, const_a)))
        elif t == "init":
            const_a = constants.get(acts[0])
            if const_a is None: continue
            formulas.append(Forall(x_var, IsFirst(x_var, const_a)))
        elif t == "responded_existence":
            const_a = constants.get(acts[0])
            const_b = constants.get(acts[1])
            if const_a is None or const_b is None: continue
            _ca = const_a.value[0].item()
            formulas.append(Forall(x_var, HasAct(x_var, const_b),
                cond_vars=[x_var],
                cond_fn=lambda xv, _s=s, _e=e, _c=_ca: (xv.value[:, _s:_e] == _c).any(dim=1),
            ))
        elif t == "coexistence":
            const_a = constants.get(acts[0])
            const_b = constants.get(acts[1])
            if const_a is None or const_b is None: continue
            formulas.append(Forall(x_var, And(
                Implies(HasAct(x_var, const_a), HasAct(x_var, const_b)),
                Implies(HasAct(x_var, const_b), HasAct(x_var, const_a)),
            )))
    return formulas
