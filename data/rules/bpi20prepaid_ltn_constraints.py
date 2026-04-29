"""
Auto-generated LTN constraints from DECLARE rules.

Dataset:   bpi20prepaid
Mode:      discriminability filter  (csv=BPI20PrepaidTravelCosts_student_model_test_declare_discriminability.csv, min_net=1)
Total:     80 constraints  (altprecedence=13, altresponse=8, altsuccession=4, chainprecedence=3, chainresponse=2, chainsuccession=3, noncoexistence=42, precedence=5)

Usage in training script
------------------------
from data.rules.bpi20prepaid_ltn_constraints import (
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
        "id": "c0003",
        "template": "precedence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Request For Payment APPROVED by ADMINISTRATION"
        ],
        "support": 0.811,
        "confidence": 1.0,
        "category": "ordering",
        "description": "'Request For Payment APPROVED by ADMINISTRATION' can only occur if 'Request For Payment SUBMITTED by EMPLOYEE' has occurred before it"
    },
    {
        "id": "c0004",
        "template": "precedence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Request For Payment APPROVED by BUDGET OWNER"
        ],
        "support": 0.3281,
        "confidence": 1.0,
        "category": "ordering",
        "description": "'Request For Payment APPROVED by BUDGET OWNER' can only occur if 'Request For Payment SUBMITTED by EMPLOYEE' has occurred before it"
    },
    {
        "id": "c0009",
        "template": "precedence",
        "arity": "binary",
        "activities": [
            "Request For Payment APPROVED by ADMINISTRATION",
            "Request For Payment APPROVED by BUDGET OWNER"
        ],
        "support": 0.3281,
        "confidence": 1.0,
        "category": "ordering",
        "description": "'Request For Payment APPROVED by BUDGET OWNER' can only occur if 'Request For Payment APPROVED by ADMINISTRATION' has occurred before it"
    },
    {
        "id": "c0015",
        "template": "precedence",
        "arity": "binary",
        "activities": [
            "Permit APPROVED by SUPERVISOR",
            "Permit FINAL_APPROVED by DIRECTOR"
        ],
        "support": 0.1793,
        "confidence": 0.9917,
        "category": "ordering",
        "description": "'Permit FINAL_APPROVED by DIRECTOR' can only occur if 'Permit APPROVED by SUPERVISOR' has occurred before it"
    },
    {
        "id": "c0017",
        "template": "precedence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Request For Payment FINAL_APPROVED by SUPERVISOR"
        ],
        "support": 0.9256,
        "confidence": 1.0,
        "category": "ordering",
        "description": "'Request For Payment FINAL_APPROVED by SUPERVISOR' can only occur if 'Request For Payment SUBMITTED by EMPLOYEE' has occurred before it"
    },
    {
        "id": "c0022",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "Permit APPROVED by BUDGET OWNER",
            "Request For Payment FINAL_APPROVED by SUPERVISOR"
        ],
        "support": 0.285,
        "confidence": 0.9138,
        "category": "ordering",
        "description": "Whenever 'Permit APPROVED by BUDGET OWNER' occurs, 'Request For Payment FINAL_APPROVED by SUPERVISOR' must follow before 'Permit APPROVED by BUDGET OWNER' can recur"
    },
    {
        "id": "c0023",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "Request For Payment APPROVED by ADMINISTRATION",
            "Payment Handled"
        ],
        "support": 0.811,
        "confidence": 0.9881,
        "category": "ordering",
        "description": "Whenever 'Request For Payment APPROVED by ADMINISTRATION' occurs, 'Payment Handled' must follow before 'Request For Payment APPROVED by ADMINISTRATION' can recur"
    },
    {
        "id": "c0025",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "Request Payment",
            "Payment Handled"
        ],
        "support": 0.9427,
        "confidence": 0.9992,
        "category": "ordering",
        "description": "Whenever 'Request Payment' occurs, 'Payment Handled' must follow before 'Request Payment' can recur"
    },
    {
        "id": "c0030",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "Permit APPROVED by BUDGET OWNER",
            "Payment Handled"
        ],
        "support": 0.285,
        "confidence": 0.9321,
        "category": "ordering",
        "description": "Whenever 'Permit APPROVED by BUDGET OWNER' occurs, 'Payment Handled' must follow before 'Permit APPROVED by BUDGET OWNER' can recur"
    },
    {
        "id": "c0031",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "Permit APPROVED by BUDGET OWNER",
            "Request Payment"
        ],
        "support": 0.285,
        "confidence": 0.9295,
        "category": "ordering",
        "description": "Whenever 'Permit APPROVED by BUDGET OWNER' occurs, 'Request Payment' must follow before 'Permit APPROVED by BUDGET OWNER' can recur"
    },
    {
        "id": "c0032",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "Permit APPROVED by BUDGET OWNER",
            "Request For Payment APPROVED by BUDGET OWNER"
        ],
        "support": 0.285,
        "confidence": 0.906,
        "category": "ordering",
        "description": "Whenever 'Permit APPROVED by BUDGET OWNER' occurs, 'Request For Payment APPROVED by BUDGET OWNER' must follow before 'Permit APPROVED by BUDGET OWNER' can recur"
    },
    {
        "id": "c0039",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "Request For Payment APPROVED by ADMINISTRATION",
            "Request Payment"
        ],
        "support": 0.811,
        "confidence": 0.9872,
        "category": "ordering",
        "description": "Whenever 'Request For Payment APPROVED by ADMINISTRATION' occurs, 'Request Payment' must follow before 'Request For Payment APPROVED by ADMINISTRATION' can recur"
    },
    {
        "id": "c0050",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "Request For Payment APPROVED by ADMINISTRATION",
            "Request For Payment FINAL_APPROVED by SUPERVISOR"
        ],
        "support": 0.811,
        "confidence": 0.9679,
        "category": "ordering",
        "description": "Whenever 'Request For Payment APPROVED by ADMINISTRATION' occurs, 'Request For Payment FINAL_APPROVED by SUPERVISOR' must follow before 'Request For Payment APPROVED by ADMINISTRATION' can recur"
    },
    {
        "id": "c0053",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "Permit APPROVED by ADMINISTRATION",
            "Permit APPROVED by BUDGET OWNER"
        ],
        "support": 0.285,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'Permit APPROVED by BUDGET OWNER' must be preceded by 'Permit APPROVED by ADMINISTRATION' with no other 'Permit APPROVED by BUDGET OWNER' in between"
    },
    {
        "id": "c0054",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "Permit APPROVED by SUPERVISOR",
            "Permit FINAL_APPROVED by DIRECTOR"
        ],
        "support": 0.1793,
        "confidence": 0.9917,
        "category": "ordering",
        "description": "Each 'Permit FINAL_APPROVED by DIRECTOR' must be preceded by 'Permit APPROVED by SUPERVISOR' with no other 'Permit FINAL_APPROVED by DIRECTOR' in between"
    },
    {
        "id": "c0056",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Request For Payment FINAL_APPROVED by SUPERVISOR"
        ],
        "support": 0.9256,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'Request For Payment FINAL_APPROVED by SUPERVISOR' must be preceded by 'Request For Payment SUBMITTED by EMPLOYEE' with no other 'Request For Payment FINAL_APPROVED by SUPERVISOR' in between"
    },
    {
        "id": "c0057",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "Request For Payment FINAL_APPROVED by SUPERVISOR",
            "Payment Handled"
        ],
        "support": 0.9435,
        "confidence": 0.9795,
        "category": "ordering",
        "description": "Each 'Payment Handled' must be preceded by 'Request For Payment FINAL_APPROVED by SUPERVISOR' with no other 'Payment Handled' in between"
    },
    {
        "id": "c0058",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Request For Payment APPROVED by ADMINISTRATION"
        ],
        "support": 0.811,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'Request For Payment APPROVED by ADMINISTRATION' must be preceded by 'Request For Payment SUBMITTED by EMPLOYEE' with no other 'Request For Payment APPROVED by ADMINISTRATION' in between"
    },
    {
        "id": "c0059",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Request For Payment APPROVED by BUDGET OWNER"
        ],
        "support": 0.3281,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'Request For Payment APPROVED by BUDGET OWNER' must be preceded by 'Request For Payment SUBMITTED by EMPLOYEE' with no other 'Request For Payment APPROVED by BUDGET OWNER' in between"
    },
    {
        "id": "c0060",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "Permit SUBMITTED by EMPLOYEE",
            "Permit APPROVED by BUDGET OWNER"
        ],
        "support": 0.285,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'Permit APPROVED by BUDGET OWNER' must be preceded by 'Permit SUBMITTED by EMPLOYEE' with no other 'Permit APPROVED by BUDGET OWNER' in between"
    },
    {
        "id": "c0061",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Request For Payment REJECTED by EMPLOYEE"
        ],
        "support": 0.1205,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'Request For Payment REJECTED by EMPLOYEE' must be preceded by 'Request For Payment SUBMITTED by EMPLOYEE' with no other 'Request For Payment REJECTED by EMPLOYEE' in between"
    },
    {
        "id": "c0063",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "Request Payment",
            "Payment Handled"
        ],
        "support": 0.9435,
        "confidence": 0.9984,
        "category": "ordering",
        "description": "Each 'Payment Handled' must be preceded by 'Request Payment' with no other 'Payment Handled' in between"
    },
    {
        "id": "c0065",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "Request For Payment APPROVED by ADMINISTRATION",
            "Request For Payment APPROVED by BUDGET OWNER"
        ],
        "support": 0.3281,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'Request For Payment APPROVED by BUDGET OWNER' must be preceded by 'Request For Payment APPROVED by ADMINISTRATION' with no other 'Request For Payment APPROVED by BUDGET OWNER' in between"
    },
    {
        "id": "c0066",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Payment Handled"
        ],
        "support": 0.9435,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'Payment Handled' must be preceded by 'Request For Payment SUBMITTED by EMPLOYEE' with no other 'Payment Handled' in between"
    },
    {
        "id": "c0067",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Request Payment"
        ],
        "support": 0.9427,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'Request Payment' must be preceded by 'Request For Payment SUBMITTED by EMPLOYEE' with no other 'Request Payment' in between"
    },
    {
        "id": "c0068",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "Request For Payment FINAL_APPROVED by SUPERVISOR",
            "Request Payment"
        ],
        "support": 0.9427,
        "confidence": 0.9795,
        "category": "ordering",
        "description": "Each 'Request Payment' must be preceded by 'Request For Payment FINAL_APPROVED by SUPERVISOR' with no other 'Request Payment' in between"
    },
    {
        "id": "c0069",
        "template": "chainprecedence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Request For Payment APPROVED by ADMINISTRATION"
        ],
        "support": 0.811,
        "confidence": 0.9954,
        "category": "immediate",
        "description": "'Request For Payment APPROVED by ADMINISTRATION' can only occur if 'Request For Payment SUBMITTED by EMPLOYEE' immediately preceded it"
    },
    {
        "id": "c0074",
        "template": "chainprecedence",
        "arity": "binary",
        "activities": [
            "Permit APPROVED by SUPERVISOR",
            "Permit FINAL_APPROVED by DIRECTOR"
        ],
        "support": 0.1793,
        "confidence": 0.9378,
        "category": "immediate",
        "description": "'Permit FINAL_APPROVED by DIRECTOR' can only occur if 'Permit APPROVED by SUPERVISOR' immediately preceded it"
    },
    {
        "id": "c0075",
        "template": "chainprecedence",
        "arity": "binary",
        "activities": [
            "Request Payment",
            "Payment Handled"
        ],
        "support": 0.9435,
        "confidence": 0.9921,
        "category": "immediate",
        "description": "'Payment Handled' can only occur if 'Request Payment' immediately preceded it"
    },
    {
        "id": "c0076",
        "template": "chainresponse",
        "arity": "binary",
        "activities": [
            "Request Payment",
            "Payment Handled"
        ],
        "support": 0.9427,
        "confidence": 0.9929,
        "category": "immediate",
        "description": "Whenever 'Request Payment' occurs, 'Payment Handled' must immediately follow"
    },
    {
        "id": "c0078",
        "template": "chainresponse",
        "arity": "binary",
        "activities": [
            "Permit APPROVED by SUPERVISOR",
            "Permit FINAL_APPROVED by DIRECTOR"
        ],
        "support": 0.1793,
        "confidence": 0.9378,
        "category": "immediate",
        "description": "Whenever 'Permit APPROVED by SUPERVISOR' occurs, 'Permit FINAL_APPROVED by DIRECTOR' must immediately follow"
    },
    {
        "id": "c0085",
        "template": "altsuccession",
        "arity": "binary",
        "activities": [
            "Request Payment",
            "Payment Handled"
        ],
        "support": 0.9435,
        "confidence": 0.9984,
        "category": "ordering",
        "description": "'Request Payment' and 'Payment Handled' alternate: each 'Request Payment' triggers exactly one 'Payment Handled' and vice versa"
    },
    {
        "id": "c0086",
        "template": "altsuccession",
        "arity": "binary",
        "activities": [
            "Request For Payment FINAL_APPROVED by SUPERVISOR",
            "Request Payment"
        ],
        "support": 0.9449,
        "confidence": 0.9772,
        "category": "ordering",
        "description": "'Request For Payment FINAL_APPROVED by SUPERVISOR' and 'Request Payment' alternate: each 'Request For Payment FINAL_APPROVED by SUPERVISOR' triggers exactly one 'Request Payment' and vice versa"
    },
    {
        "id": "c0087",
        "template": "altsuccession",
        "arity": "binary",
        "activities": [
            "Request For Payment FINAL_APPROVED by SUPERVISOR",
            "Payment Handled"
        ],
        "support": 0.9449,
        "confidence": 0.978,
        "category": "ordering",
        "description": "'Request For Payment FINAL_APPROVED by SUPERVISOR' and 'Payment Handled' alternate: each 'Request For Payment FINAL_APPROVED by SUPERVISOR' triggers exactly one 'Payment Handled' and vice versa"
    },
    {
        "id": "c0088",
        "template": "altsuccession",
        "arity": "binary",
        "activities": [
            "Permit APPROVED by SUPERVISOR",
            "Permit FINAL_APPROVED by DIRECTOR"
        ],
        "support": 0.1793,
        "confidence": 0.9917,
        "category": "ordering",
        "description": "'Permit APPROVED by SUPERVISOR' and 'Permit FINAL_APPROVED by DIRECTOR' alternate: each 'Permit APPROVED by SUPERVISOR' triggers exactly one 'Permit FINAL_APPROVED by DIRECTOR' and vice versa"
    },
    {
        "id": "c0089",
        "template": "chainsuccession",
        "arity": "binary",
        "activities": [
            "Request Payment",
            "Payment Handled"
        ],
        "support": 0.9435,
        "confidence": 0.9921,
        "category": "immediate",
        "description": "'Request Payment' and 'Payment Handled' must always occur as consecutive pairs"
    },
    {
        "id": "c0090",
        "template": "chainsuccession",
        "arity": "binary",
        "activities": [
            "Request For Payment FINAL_APPROVED by SUPERVISOR",
            "Request Payment"
        ],
        "support": 0.9449,
        "confidence": 0.9567,
        "category": "immediate",
        "description": "'Request For Payment FINAL_APPROVED by SUPERVISOR' and 'Request Payment' must always occur as consecutive pairs"
    },
    {
        "id": "c0091",
        "template": "chainsuccession",
        "arity": "binary",
        "activities": [
            "Permit APPROVED by SUPERVISOR",
            "Permit FINAL_APPROVED by DIRECTOR"
        ],
        "support": 0.1793,
        "confidence": 0.9378,
        "category": "immediate",
        "description": "'Permit APPROVED by SUPERVISOR' and 'Permit FINAL_APPROVED by DIRECTOR' must always occur as consecutive pairs"
    },
    {
        "id": "c0164",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit FINAL_APPROVED by DIRECTOR",
            "Permit REJECTED by EMPLOYEE"
        ],
        "support": 0.2121,
        "confidence": 0.9825,
        "category": "cross_path",
        "description": "'Permit FINAL_APPROVED by DIRECTOR' and 'Permit REJECTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0166",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit FINAL_APPROVED by DIRECTOR",
            "Permit FINAL_APPROVED by SUPERVISOR"
        ],
        "support": 0.8869,
        "confidence": 0.9983,
        "category": "cross_path",
        "description": "'Permit FINAL_APPROVED by DIRECTOR' and 'Permit FINAL_APPROVED by SUPERVISOR' cannot both occur in the same trace"
    },
    {
        "id": "c0167",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit FINAL_APPROVED by DIRECTOR",
            "Permit REJECTED by BUDGET OWNER"
        ],
        "support": 0.1853,
        "confidence": 1.0,
        "category": "cross_path",
        "description": "'Permit FINAL_APPROVED by DIRECTOR' and 'Permit REJECTED by BUDGET OWNER' cannot both occur in the same trace"
    },
    {
        "id": "c0169",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit FINAL_APPROVED by DIRECTOR",
            "Permit REJECTED by ADMINISTRATION"
        ],
        "support": 0.1853,
        "confidence": 0.996,
        "category": "cross_path",
        "description": "'Permit FINAL_APPROVED by DIRECTOR' and 'Permit REJECTED by ADMINISTRATION' cannot both occur in the same trace"
    },
    {
        "id": "c0171",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit FINAL_APPROVED by DIRECTOR",
            "Request For Payment SAVED by EMPLOYEE"
        ],
        "support": 0.1912,
        "confidence": 0.9883,
        "category": "cross_path",
        "description": "'Permit FINAL_APPROVED by DIRECTOR' and 'Request For Payment SAVED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0181",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment REJECTED by SUPERVISOR",
            "Request For Payment FINAL_APPROVED by SUPERVISOR"
        ],
        "support": 0.9286,
        "confidence": 0.996,
        "category": "cross_path",
        "description": "'Request For Payment REJECTED by SUPERVISOR' and 'Request For Payment FINAL_APPROVED by SUPERVISOR' cannot both occur in the same trace"
    },
    {
        "id": "c0184",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment REJECTED by SUPERVISOR",
            "Request For Payment APPROVED by BUDGET OWNER"
        ],
        "support": 0.3311,
        "confidence": 0.9888,
        "category": "cross_path",
        "description": "'Request For Payment REJECTED by SUPERVISOR' and 'Request For Payment APPROVED by BUDGET OWNER' cannot both occur in the same trace"
    },
    {
        "id": "c0195",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment FINAL_APPROVED by SUPERVISOR",
            "Request For Payment REJECTED by SUPERVISOR"
        ],
        "support": 0.9286,
        "confidence": 0.996,
        "category": "cross_path",
        "description": "'Request For Payment FINAL_APPROVED by SUPERVISOR' and 'Request For Payment REJECTED by SUPERVISOR' cannot both occur in the same trace"
    },
    {
        "id": "c0203",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment FINAL_APPROVED by SUPERVISOR",
            "Permit REJECTED by ADMINISTRATION"
        ],
        "support": 0.9278,
        "confidence": 0.9952,
        "category": "cross_path",
        "description": "'Request For Payment FINAL_APPROVED by SUPERVISOR' and 'Permit REJECTED by ADMINISTRATION' cannot both occur in the same trace"
    },
    {
        "id": "c0219",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment APPROVED by ADMINISTRATION",
            "Permit REJECTED by EMPLOYEE"
        ],
        "support": 0.8229,
        "confidence": 0.9702,
        "category": "cross_path",
        "description": "'Request For Payment APPROVED by ADMINISTRATION' and 'Permit REJECTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0220",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment APPROVED by ADMINISTRATION",
            "Permit REJECTED by BUDGET OWNER"
        ],
        "support": 0.8118,
        "confidence": 0.9936,
        "category": "cross_path",
        "description": "'Request For Payment APPROVED by ADMINISTRATION' and 'Permit REJECTED by BUDGET OWNER' cannot both occur in the same trace"
    },
    {
        "id": "c0222",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment APPROVED by ADMINISTRATION",
            "Permit REJECTED by ADMINISTRATION"
        ],
        "support": 0.8118,
        "confidence": 0.9927,
        "category": "cross_path",
        "description": "'Request For Payment APPROVED by ADMINISTRATION' and 'Permit REJECTED by ADMINISTRATION' cannot both occur in the same trace"
    },
    {
        "id": "c0229",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment APPROVED by ADMINISTRATION",
            "Permit REJECTED by SUPERVISOR"
        ],
        "support": 0.8147,
        "confidence": 0.9817,
        "category": "cross_path",
        "description": "'Request For Payment APPROVED by ADMINISTRATION' and 'Permit REJECTED by SUPERVISOR' cannot both occur in the same trace"
    },
    {
        "id": "c0266",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment APPROVED by BUDGET OWNER",
            "Request For Payment REJECTED by SUPERVISOR"
        ],
        "support": 0.3311,
        "confidence": 0.9888,
        "category": "cross_path",
        "description": "'Request For Payment APPROVED by BUDGET OWNER' and 'Request For Payment REJECTED by SUPERVISOR' cannot both occur in the same trace"
    },
    {
        "id": "c0271",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment APPROVED by BUDGET OWNER",
            "Permit REJECTED by EMPLOYEE"
        ],
        "support": 0.3519,
        "confidence": 0.9641,
        "category": "cross_path",
        "description": "'Request For Payment APPROVED by BUDGET OWNER' and 'Permit REJECTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0274",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment APPROVED by BUDGET OWNER",
            "Permit REJECTED by ADMINISTRATION"
        ],
        "support": 0.3318,
        "confidence": 0.991,
        "category": "cross_path",
        "description": "'Request For Payment APPROVED by BUDGET OWNER' and 'Permit REJECTED by ADMINISTRATION' cannot both occur in the same trace"
    },
    {
        "id": "c0326",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by EMPLOYEE",
            "Permit FINAL_APPROVED by DIRECTOR"
        ],
        "support": 0.2121,
        "confidence": 0.9825,
        "category": "cross_path",
        "description": "'Permit REJECTED by EMPLOYEE' and 'Permit FINAL_APPROVED by DIRECTOR' cannot both occur in the same trace"
    },
    {
        "id": "c0328",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by EMPLOYEE",
            "Request For Payment APPROVED by ADMINISTRATION"
        ],
        "support": 0.8229,
        "confidence": 0.9702,
        "category": "cross_path",
        "description": "'Permit REJECTED by EMPLOYEE' and 'Request For Payment APPROVED by ADMINISTRATION' cannot both occur in the same trace"
    },
    {
        "id": "c0330",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by EMPLOYEE",
            "Request For Payment APPROVED by BUDGET OWNER"
        ],
        "support": 0.3519,
        "confidence": 0.9641,
        "category": "cross_path",
        "description": "'Permit REJECTED by EMPLOYEE' and 'Request For Payment APPROVED by BUDGET OWNER' cannot both occur in the same trace"
    },
    {
        "id": "c0334",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by EMPLOYEE",
            "Request For Payment SUBMITTED by EMPLOYEE"
        ],
        "support": 0.9874,
        "confidence": 0.9631,
        "category": "cross_path",
        "description": "'Permit REJECTED by EMPLOYEE' and 'Request For Payment SUBMITTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0360",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit FINAL_APPROVED by SUPERVISOR",
            "Permit FINAL_APPROVED by DIRECTOR"
        ],
        "support": 0.8869,
        "confidence": 0.9983,
        "category": "cross_path",
        "description": "'Permit FINAL_APPROVED by SUPERVISOR' and 'Permit FINAL_APPROVED by DIRECTOR' cannot both occur in the same trace"
    },
    {
        "id": "c0370",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit FINAL_APPROVED by SUPERVISOR",
            "Request For Payment REJECTED by ADMINISTRATION"
        ],
        "support": 0.7374,
        "confidence": 0.8991,
        "category": "cross_path",
        "description": "'Permit FINAL_APPROVED by SUPERVISOR' and 'Request For Payment REJECTED by ADMINISTRATION' cannot both occur in the same trace"
    },
    {
        "id": "c0376",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit FINAL_APPROVED by SUPERVISOR",
            "Permit REJECTED by SUPERVISOR"
        ],
        "support": 0.7135,
        "confidence": 0.9802,
        "category": "cross_path",
        "description": "'Permit FINAL_APPROVED by SUPERVISOR' and 'Permit REJECTED by SUPERVISOR' cannot both occur in the same trace"
    },
    {
        "id": "c0379",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by BUDGET OWNER",
            "Permit FINAL_APPROVED by DIRECTOR"
        ],
        "support": 0.1853,
        "confidence": 1.0,
        "category": "cross_path",
        "description": "'Permit REJECTED by BUDGET OWNER' and 'Permit FINAL_APPROVED by DIRECTOR' cannot both occur in the same trace"
    },
    {
        "id": "c0381",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by BUDGET OWNER",
            "Request For Payment APPROVED by ADMINISTRATION"
        ],
        "support": 0.8118,
        "confidence": 0.9936,
        "category": "cross_path",
        "description": "'Permit REJECTED by BUDGET OWNER' and 'Request For Payment APPROVED by ADMINISTRATION' cannot both occur in the same trace"
    },
    {
        "id": "c0387",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by BUDGET OWNER",
            "Request For Payment SUBMITTED by EMPLOYEE"
        ],
        "support": 0.9874,
        "confidence": 0.994,
        "category": "cross_path",
        "description": "'Permit REJECTED by BUDGET OWNER' and 'Request For Payment SUBMITTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0407",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by ADMINISTRATION",
            "Permit FINAL_APPROVED by DIRECTOR"
        ],
        "support": 0.1853,
        "confidence": 0.996,
        "category": "cross_path",
        "description": "'Permit REJECTED by ADMINISTRATION' and 'Permit FINAL_APPROVED by DIRECTOR' cannot both occur in the same trace"
    },
    {
        "id": "c0408",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by ADMINISTRATION",
            "Request For Payment FINAL_APPROVED by SUPERVISOR"
        ],
        "support": 0.9278,
        "confidence": 0.9952,
        "category": "cross_path",
        "description": "'Permit REJECTED by ADMINISTRATION' and 'Request For Payment FINAL_APPROVED by SUPERVISOR' cannot both occur in the same trace"
    },
    {
        "id": "c0409",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by ADMINISTRATION",
            "Request For Payment APPROVED by ADMINISTRATION"
        ],
        "support": 0.8118,
        "confidence": 0.9927,
        "category": "cross_path",
        "description": "'Permit REJECTED by ADMINISTRATION' and 'Request For Payment APPROVED by ADMINISTRATION' cannot both occur in the same trace"
    },
    {
        "id": "c0411",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by ADMINISTRATION",
            "Request For Payment APPROVED by BUDGET OWNER"
        ],
        "support": 0.3318,
        "confidence": 0.991,
        "category": "cross_path",
        "description": "'Permit REJECTED by ADMINISTRATION' and 'Request For Payment APPROVED by BUDGET OWNER' cannot both occur in the same trace"
    },
    {
        "id": "c0415",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by ADMINISTRATION",
            "Request For Payment SUBMITTED by EMPLOYEE"
        ],
        "support": 0.9874,
        "confidence": 0.9932,
        "category": "cross_path",
        "description": "'Permit REJECTED by ADMINISTRATION' and 'Request For Payment SUBMITTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0433",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment REJECTED by ADMINISTRATION",
            "Permit FINAL_APPROVED by SUPERVISOR"
        ],
        "support": 0.7374,
        "confidence": 0.8991,
        "category": "cross_path",
        "description": "'Request For Payment REJECTED by ADMINISTRATION' and 'Permit FINAL_APPROVED by SUPERVISOR' cannot both occur in the same trace"
    },
    {
        "id": "c0448",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment SAVED by EMPLOYEE",
            "Permit FINAL_APPROVED by DIRECTOR"
        ],
        "support": 0.1912,
        "confidence": 0.9883,
        "category": "cross_path",
        "description": "'Request For Payment SAVED by EMPLOYEE' and 'Permit FINAL_APPROVED by DIRECTOR' cannot both occur in the same trace"
    },
    {
        "id": "c0456",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment SAVED by EMPLOYEE",
            "Request For Payment SUBMITTED by EMPLOYEE"
        ],
        "support": 1.0,
        "confidence": 0.9985,
        "category": "cross_path",
        "description": "'Request For Payment SAVED by EMPLOYEE' and 'Request For Payment SUBMITTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0468",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Permit REJECTED by EMPLOYEE"
        ],
        "support": 0.9874,
        "confidence": 0.9631,
        "category": "cross_path",
        "description": "'Request For Payment SUBMITTED by EMPLOYEE' and 'Permit REJECTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0469",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Permit REJECTED by BUDGET OWNER"
        ],
        "support": 0.9874,
        "confidence": 0.994,
        "category": "cross_path",
        "description": "'Request For Payment SUBMITTED by EMPLOYEE' and 'Permit REJECTED by BUDGET OWNER' cannot both occur in the same trace"
    },
    {
        "id": "c0471",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Permit REJECTED by ADMINISTRATION"
        ],
        "support": 0.9874,
        "confidence": 0.9932,
        "category": "cross_path",
        "description": "'Request For Payment SUBMITTED by EMPLOYEE' and 'Permit REJECTED by ADMINISTRATION' cannot both occur in the same trace"
    },
    {
        "id": "c0473",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Request For Payment SAVED by EMPLOYEE"
        ],
        "support": 1.0,
        "confidence": 0.9985,
        "category": "cross_path",
        "description": "'Request For Payment SUBMITTED by EMPLOYEE' and 'Request For Payment SAVED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0478",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Permit REJECTED by SUPERVISOR"
        ],
        "support": 0.9874,
        "confidence": 0.9812,
        "category": "cross_path",
        "description": "'Request For Payment SUBMITTED by EMPLOYEE' and 'Permit REJECTED by SUPERVISOR' cannot both occur in the same trace"
    },
    {
        "id": "c0588",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by SUPERVISOR",
            "Request For Payment APPROVED by ADMINISTRATION"
        ],
        "support": 0.8147,
        "confidence": 0.9817,
        "category": "cross_path",
        "description": "'Permit REJECTED by SUPERVISOR' and 'Request For Payment APPROVED by ADMINISTRATION' cannot both occur in the same trace"
    },
    {
        "id": "c0592",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by SUPERVISOR",
            "Permit FINAL_APPROVED by SUPERVISOR"
        ],
        "support": 0.7135,
        "confidence": 0.9802,
        "category": "cross_path",
        "description": "'Permit REJECTED by SUPERVISOR' and 'Permit FINAL_APPROVED by SUPERVISOR' cannot both occur in the same trace"
    },
    {
        "id": "c0594",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by SUPERVISOR",
            "Request For Payment SUBMITTED by EMPLOYEE"
        ],
        "support": 0.9874,
        "confidence": 0.9812,
        "category": "cross_path",
        "description": "'Permit REJECTED by SUPERVISOR' and 'Request For Payment SUBMITTED by EMPLOYEE' cannot both occur in the same trace"
    }
]

# Ordered list of all unique activities referenced by the constraints above
UNIQUE_ACTIVITIES: list[str] = ['Request For Payment SUBMITTED by EMPLOYEE', 'Request For Payment APPROVED by ADMINISTRATION', 'Request For Payment APPROVED by BUDGET OWNER', 'Permit APPROVED by SUPERVISOR', 'Permit FINAL_APPROVED by DIRECTOR', 'Request For Payment FINAL_APPROVED by SUPERVISOR', 'Permit APPROVED by BUDGET OWNER', 'Payment Handled', 'Request Payment', 'Permit APPROVED by ADMINISTRATION', 'Permit SUBMITTED by EMPLOYEE', 'Request For Payment REJECTED by EMPLOYEE', 'Permit REJECTED by EMPLOYEE', 'Permit FINAL_APPROVED by SUPERVISOR', 'Permit REJECTED by BUDGET OWNER', 'Permit REJECTED by ADMINISTRATION', 'Request For Payment SAVED by EMPLOYEE', 'Request For Payment REJECTED by SUPERVISOR', 'Permit REJECTED by SUPERVISOR', 'Request For Payment REJECTED by ADMINISTRATION']


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
