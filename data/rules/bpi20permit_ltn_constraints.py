"""
Auto-generated LTN constraints from DECLARE rules.

Dataset:   bpi20permit
Mode:      discriminability filter  (csv=BPI20TravelPermitData_student_model_val_declare_discriminability.csv, min_net=1)
Total:     61 constraints  (altprecedence=6, altresponse=4, altsuccession=2, chainprecedence=2, chainresponse=4, chainsuccession=1, exactly_one=1, noncoexistence=38, precedence=3)

Usage in training script
------------------------
from data.rules.bpi20permit_ltn_constraints import (
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
        "template": "precedence",
        "arity": "binary",
        "activities": [
            "Permit SUBMITTED by EMPLOYEE",
            "Declaration SUBMITTED by EMPLOYEE"
        ],
        "support": 0.7908,
        "confidence": 1.0,
        "category": "ordering",
        "description": "'Declaration SUBMITTED by EMPLOYEE' can only occur if 'Permit SUBMITTED by EMPLOYEE' has occurred before it"
    },
    {
        "id": "c0018",
        "template": "precedence",
        "arity": "binary",
        "activities": [
            "End trip",
            "Declaration APPROVED by ADMINISTRATION"
        ],
        "support": 0.6327,
        "confidence": 0.9689,
        "category": "ordering",
        "description": "'Declaration APPROVED by ADMINISTRATION' can only occur if 'End trip' has occurred before it"
    },
    {
        "id": "c0047",
        "template": "precedence",
        "arity": "binary",
        "activities": [
            "Start trip",
            "Declaration APPROVED by ADMINISTRATION"
        ],
        "support": 0.6327,
        "confidence": 0.9899,
        "category": "ordering",
        "description": "'Declaration APPROVED by ADMINISTRATION' can only occur if 'Start trip' has occurred before it"
    },
    {
        "id": "c0053",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "End trip",
            "Declaration APPROVED by BUDGET OWNER"
        ],
        "support": 0.226,
        "confidence": 0.9364,
        "category": "ordering",
        "description": "Each 'Declaration APPROVED by BUDGET OWNER' must be preceded by 'End trip' with no other 'Declaration APPROVED by BUDGET OWNER' in between"
    },
    {
        "id": "c0058",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "Request Payment",
            "Payment Handled"
        ],
        "support": 0.8111,
        "confidence": 0.9602,
        "category": "ordering",
        "description": "Each 'Payment Handled' must be preceded by 'Request Payment' with no other 'Payment Handled' in between"
    },
    {
        "id": "c0061",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "Declaration SUBMITTED by EMPLOYEE",
            "Declaration APPROVED by BUDGET OWNER"
        ],
        "support": 0.226,
        "confidence": 0.9863,
        "category": "ordering",
        "description": "Each 'Declaration APPROVED by BUDGET OWNER' must be preceded by 'Declaration SUBMITTED by EMPLOYEE' with no other 'Declaration APPROVED by BUDGET OWNER' in between"
    },
    {
        "id": "c0065",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "Declaration APPROVED by ADMINISTRATION",
            "Declaration APPROVED by BUDGET OWNER"
        ],
        "support": 0.226,
        "confidence": 0.9873,
        "category": "ordering",
        "description": "Each 'Declaration APPROVED by BUDGET OWNER' must be preceded by 'Declaration APPROVED by ADMINISTRATION' with no other 'Declaration APPROVED by BUDGET OWNER' in between"
    },
    {
        "id": "c0074",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "Start trip",
            "Declaration APPROVED by BUDGET OWNER"
        ],
        "support": 0.226,
        "confidence": 0.956,
        "category": "ordering",
        "description": "Each 'Declaration APPROVED by BUDGET OWNER' must be preceded by 'Start trip' with no other 'Declaration APPROVED by BUDGET OWNER' in between"
    },
    {
        "id": "c0075",
        "template": "altprecedence",
        "arity": "binary",
        "activities": [
            "Start trip",
            "End trip"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "category": "ordering",
        "description": "Each 'End trip' must be preceded by 'Start trip' with no other 'End trip' in between"
    },
    {
        "id": "c0081",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "Declaration APPROVED by BUDGET OWNER",
            "Declaration FINAL_APPROVED by SUPERVISOR"
        ],
        "support": 0.226,
        "confidence": 0.9442,
        "category": "ordering",
        "description": "Whenever 'Declaration APPROVED by BUDGET OWNER' occurs, 'Declaration FINAL_APPROVED by SUPERVISOR' must follow before 'Declaration APPROVED by BUDGET OWNER' can recur"
    },
    {
        "id": "c0082",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "Declaration APPROVED by BUDGET OWNER",
            "Request Payment"
        ],
        "support": 0.226,
        "confidence": 0.9726,
        "category": "ordering",
        "description": "Whenever 'Declaration APPROVED by BUDGET OWNER' occurs, 'Request Payment' must follow before 'Declaration APPROVED by BUDGET OWNER' can recur"
    },
    {
        "id": "c0083",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "Request Payment",
            "Payment Handled"
        ],
        "support": 0.8109,
        "confidence": 0.9607,
        "category": "ordering",
        "description": "Whenever 'Request Payment' occurs, 'Payment Handled' must follow before 'Request Payment' can recur"
    },
    {
        "id": "c0088",
        "template": "altresponse",
        "arity": "binary",
        "activities": [
            "Declaration APPROVED by BUDGET OWNER",
            "Payment Handled"
        ],
        "support": 0.226,
        "confidence": 0.9726,
        "category": "ordering",
        "description": "Whenever 'Declaration APPROVED by BUDGET OWNER' occurs, 'Payment Handled' must follow before 'Declaration APPROVED by BUDGET OWNER' can recur"
    },
    {
        "id": "c0094",
        "template": "chainprecedence",
        "arity": "binary",
        "activities": [
            "Declaration APPROVED by ADMINISTRATION",
            "Declaration APPROVED by BUDGET OWNER"
        ],
        "support": 0.226,
        "confidence": 0.9736,
        "category": "immediate",
        "description": "'Declaration APPROVED by BUDGET OWNER' can only occur if 'Declaration APPROVED by ADMINISTRATION' immediately preceded it"
    },
    {
        "id": "c0095",
        "template": "chainprecedence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Request For Payment APPROVED by ADMINISTRATION"
        ],
        "support": 0.151,
        "confidence": 0.9824,
        "category": "immediate",
        "description": "'Request For Payment APPROVED by ADMINISTRATION' can only occur if 'Request For Payment SUBMITTED by EMPLOYEE' immediately preceded it"
    },
    {
        "id": "c0097",
        "template": "chainresponse",
        "arity": "binary",
        "activities": [
            "Declaration REJECTED by EMPLOYEE",
            "Declaration SUBMITTED by EMPLOYEE"
        ],
        "support": 0.197,
        "confidence": 0.927,
        "category": "immediate",
        "description": "Whenever 'Declaration REJECTED by EMPLOYEE' occurs, 'Declaration SUBMITTED by EMPLOYEE' must immediately follow"
    },
    {
        "id": "c0098",
        "template": "chainresponse",
        "arity": "binary",
        "activities": [
            "Declaration APPROVED by BUDGET OWNER",
            "Declaration FINAL_APPROVED by SUPERVISOR"
        ],
        "support": 0.226,
        "confidence": 0.9247,
        "category": "immediate",
        "description": "Whenever 'Declaration APPROVED by BUDGET OWNER' occurs, 'Declaration FINAL_APPROVED by SUPERVISOR' must immediately follow"
    },
    {
        "id": "c0100",
        "template": "chainresponse",
        "arity": "binary",
        "activities": [
            "Declaration FINAL_APPROVED by SUPERVISOR",
            "Request Payment"
        ],
        "support": 0.753,
        "confidence": 0.9653,
        "category": "immediate",
        "description": "Whenever 'Declaration FINAL_APPROVED by SUPERVISOR' occurs, 'Request Payment' must immediately follow"
    },
    {
        "id": "c0101",
        "template": "chainresponse",
        "arity": "binary",
        "activities": [
            "Request Payment",
            "Payment Handled"
        ],
        "support": 0.8109,
        "confidence": 0.9376,
        "category": "immediate",
        "description": "Whenever 'Request Payment' occurs, 'Payment Handled' must immediately follow"
    },
    {
        "id": "c0102",
        "template": "exactly_one",
        "arity": "unary",
        "activities": [
            "End trip"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "category": "occurrence",
        "description": "'End trip' must occur exactly once"
    },
    {
        "id": "c0119",
        "template": "altsuccession",
        "arity": "binary",
        "activities": [
            "Request Payment",
            "Payment Handled"
        ],
        "support": 0.8111,
        "confidence": 0.9602,
        "category": "ordering",
        "description": "'Request Payment' and 'Payment Handled' alternate: each 'Request Payment' triggers exactly one 'Payment Handled' and vice versa"
    },
    {
        "id": "c0120",
        "template": "altsuccession",
        "arity": "binary",
        "activities": [
            "Start trip",
            "End trip"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "category": "ordering",
        "description": "'Start trip' and 'End trip' alternate: each 'Start trip' triggers exactly one 'End trip' and vice versa"
    },
    {
        "id": "c0121",
        "template": "chainsuccession",
        "arity": "binary",
        "activities": [
            "Request Payment",
            "Payment Handled"
        ],
        "support": 0.8111,
        "confidence": 0.937,
        "category": "immediate",
        "description": "'Request Payment' and 'Payment Handled' must always occur as consecutive pairs"
    },
    {
        "id": "c0176",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Send Reminder",
            "Request For Payment APPROVED by ADMINISTRATION"
        ],
        "support": 0.3299,
        "confidence": 0.9484,
        "category": "cross_path",
        "description": "'Send Reminder' and 'Request For Payment APPROVED by ADMINISTRATION' cannot both occur in the same trace"
    },
    {
        "id": "c0197",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Send Reminder",
            "Declaration SUBMITTED by EMPLOYEE"
        ],
        "support": 0.9308,
        "confidence": 0.9399,
        "category": "cross_path",
        "description": "'Send Reminder' and 'Declaration SUBMITTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0249",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit FINAL_APPROVED by DIRECTOR",
            "Request For Payment SUBMITTED by EMPLOYEE"
        ],
        "support": 0.2508,
        "confidence": 0.8757,
        "category": "cross_path",
        "description": "'Permit FINAL_APPROVED by DIRECTOR' and 'Request For Payment SUBMITTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0287",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Declaration REJECTED by ADMINISTRATION",
            "Request For Payment SUBMITTED by EMPLOYEE"
        ],
        "support": 0.3235,
        "confidence": 0.892,
        "category": "cross_path",
        "description": "'Declaration REJECTED by ADMINISTRATION' and 'Request For Payment SUBMITTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0317",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Declaration REJECTED by ADMINISTRATION",
            "Request For Payment APPROVED by BUDGET OWNER"
        ],
        "support": 0.2141,
        "confidence": 0.9298,
        "category": "cross_path",
        "description": "'Declaration REJECTED by ADMINISTRATION' and 'Request For Payment APPROVED by BUDGET OWNER' cannot both occur in the same trace"
    },
    {
        "id": "c0326",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit APPROVED by BUDGET OWNER",
            "Permit REJECTED by SUPERVISOR"
        ],
        "support": 0.2963,
        "confidence": 0.9761,
        "category": "cross_path",
        "description": "'Permit APPROVED by BUDGET OWNER' and 'Permit REJECTED by SUPERVISOR' cannot both occur in the same trace"
    },
    {
        "id": "c0327",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit APPROVED by BUDGET OWNER",
            "Request For Payment SUBMITTED by EMPLOYEE"
        ],
        "support": 0.4138,
        "confidence": 0.8701,
        "category": "cross_path",
        "description": "'Permit APPROVED by BUDGET OWNER' and 'Request For Payment SUBMITTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0328",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit APPROVED by BUDGET OWNER",
            "Permit REJECTED by EMPLOYEE"
        ],
        "support": 0.3156,
        "confidence": 0.9467,
        "category": "cross_path",
        "description": "'Permit APPROVED by BUDGET OWNER' and 'Permit REJECTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0338",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit APPROVED by BUDGET OWNER",
            "Permit REJECTED by ADMINISTRATION"
        ],
        "support": 0.2977,
        "confidence": 0.9799,
        "category": "cross_path",
        "description": "'Permit APPROVED by BUDGET OWNER' and 'Permit REJECTED by ADMINISTRATION' cannot both occur in the same trace"
    },
    {
        "id": "c0351",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit APPROVED by BUDGET OWNER",
            "Request For Payment REJECTED by ADMINISTRATION"
        ],
        "support": 0.3003,
        "confidence": 0.9691,
        "category": "cross_path",
        "description": "'Permit APPROVED by BUDGET OWNER' and 'Request For Payment REJECTED by ADMINISTRATION' cannot both occur in the same trace"
    },
    {
        "id": "c0437",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by SUPERVISOR",
            "Permit APPROVED by BUDGET OWNER"
        ],
        "support": 0.2963,
        "confidence": 0.9761,
        "category": "cross_path",
        "description": "'Permit REJECTED by SUPERVISOR' and 'Permit APPROVED by BUDGET OWNER' cannot both occur in the same trace"
    },
    {
        "id": "c0445",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by SUPERVISOR",
            "Permit FINAL_APPROVED by SUPERVISOR"
        ],
        "support": 0.8919,
        "confidence": 0.9854,
        "category": "cross_path",
        "description": "'Permit REJECTED by SUPERVISOR' and 'Permit FINAL_APPROVED by SUPERVISOR' cannot both occur in the same trace"
    },
    {
        "id": "c0449",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by SUPERVISOR",
            "Start trip"
        ],
        "support": 1.0,
        "confidence": 0.981,
        "category": "cross_path",
        "description": "'Permit REJECTED by SUPERVISOR' and 'Start trip' cannot both occur in the same trace"
    },
    {
        "id": "c0457",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Permit FINAL_APPROVED by DIRECTOR"
        ],
        "support": 0.2508,
        "confidence": 0.8757,
        "category": "cross_path",
        "description": "'Request For Payment SUBMITTED by EMPLOYEE' and 'Permit FINAL_APPROVED by DIRECTOR' cannot both occur in the same trace"
    },
    {
        "id": "c0458",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Declaration REJECTED by ADMINISTRATION"
        ],
        "support": 0.3235,
        "confidence": 0.892,
        "category": "cross_path",
        "description": "'Request For Payment SUBMITTED by EMPLOYEE' and 'Declaration REJECTED by ADMINISTRATION' cannot both occur in the same trace"
    },
    {
        "id": "c0459",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Permit APPROVED by BUDGET OWNER"
        ],
        "support": 0.4138,
        "confidence": 0.8701,
        "category": "cross_path",
        "description": "'Request For Payment SUBMITTED by EMPLOYEE' and 'Permit APPROVED by BUDGET OWNER' cannot both occur in the same trace"
    },
    {
        "id": "c0461",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Permit REJECTED by EMPLOYEE"
        ],
        "support": 0.224,
        "confidence": 0.9684,
        "category": "cross_path",
        "description": "'Request For Payment SUBMITTED by EMPLOYEE' and 'Permit REJECTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0464",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Permit SUBMITTED by EMPLOYEE"
        ],
        "support": 0.9976,
        "confidence": 0.8164,
        "category": "cross_path",
        "description": "'Request For Payment SUBMITTED by EMPLOYEE' and 'Permit SUBMITTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0472",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Permit REJECTED by ADMINISTRATION"
        ],
        "support": 0.2008,
        "confidence": 0.9923,
        "category": "cross_path",
        "description": "'Request For Payment SUBMITTED by EMPLOYEE' and 'Permit REJECTED by ADMINISTRATION' cannot both occur in the same trace"
    },
    {
        "id": "c0475",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Permit FINAL_APPROVED by SUPERVISOR"
        ],
        "support": 0.9171,
        "confidence": 0.8343,
        "category": "cross_path",
        "description": "'Request For Payment SUBMITTED by EMPLOYEE' and 'Permit FINAL_APPROVED by SUPERVISOR' cannot both occur in the same trace"
    },
    {
        "id": "c0481",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Permit APPROVED by ADMINISTRATION"
        ],
        "support": 0.8238,
        "confidence": 0.8164,
        "category": "cross_path",
        "description": "'Request For Payment SUBMITTED by EMPLOYEE' and 'Permit APPROVED by ADMINISTRATION' cannot both occur in the same trace"
    },
    {
        "id": "c0489",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment SUBMITTED by EMPLOYEE",
            "Permit APPROVED by SUPERVISOR"
        ],
        "support": 0.2508,
        "confidence": 0.8757,
        "category": "cross_path",
        "description": "'Request For Payment SUBMITTED by EMPLOYEE' and 'Permit APPROVED by SUPERVISOR' cannot both occur in the same trace"
    },
    {
        "id": "c0500",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by EMPLOYEE",
            "Permit APPROVED by BUDGET OWNER"
        ],
        "support": 0.3156,
        "confidence": 0.9467,
        "category": "cross_path",
        "description": "'Permit REJECTED by EMPLOYEE' and 'Permit APPROVED by BUDGET OWNER' cannot both occur in the same trace"
    },
    {
        "id": "c0503",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by EMPLOYEE",
            "Request For Payment SUBMITTED by EMPLOYEE"
        ],
        "support": 0.224,
        "confidence": 0.9684,
        "category": "cross_path",
        "description": "'Permit REJECTED by EMPLOYEE' and 'Request For Payment SUBMITTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0563",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit SUBMITTED by EMPLOYEE",
            "Request For Payment SUBMITTED by EMPLOYEE"
        ],
        "support": 0.9976,
        "confidence": 0.8164,
        "category": "cross_path",
        "description": "'Permit SUBMITTED by EMPLOYEE' and 'Request For Payment SUBMITTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0669",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit SAVED by EMPLOYEE",
            "Declaration SUBMITTED by EMPLOYEE"
        ],
        "support": 0.7935,
        "confidence": 1.0,
        "category": "cross_path",
        "description": "'Permit SAVED by EMPLOYEE' and 'Declaration SUBMITTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0692",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment APPROVED by ADMINISTRATION",
            "Send Reminder"
        ],
        "support": 0.3299,
        "confidence": 0.9484,
        "category": "cross_path",
        "description": "'Request For Payment APPROVED by ADMINISTRATION' and 'Send Reminder' cannot both occur in the same trace"
    },
    {
        "id": "c0793",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by ADMINISTRATION",
            "Permit APPROVED by BUDGET OWNER"
        ],
        "support": 0.2977,
        "confidence": 0.9799,
        "category": "cross_path",
        "description": "'Permit REJECTED by ADMINISTRATION' and 'Permit APPROVED by BUDGET OWNER' cannot both occur in the same trace"
    },
    {
        "id": "c0796",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit REJECTED by ADMINISTRATION",
            "Request For Payment SUBMITTED by EMPLOYEE"
        ],
        "support": 0.2008,
        "confidence": 0.9923,
        "category": "cross_path",
        "description": "'Permit REJECTED by ADMINISTRATION' and 'Request For Payment SUBMITTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c0872",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit FINAL_APPROVED by SUPERVISOR",
            "Permit REJECTED by SUPERVISOR"
        ],
        "support": 0.8919,
        "confidence": 0.9854,
        "category": "cross_path",
        "description": "'Permit FINAL_APPROVED by SUPERVISOR' and 'Permit REJECTED by SUPERVISOR' cannot both occur in the same trace"
    },
    {
        "id": "c0873",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit FINAL_APPROVED by SUPERVISOR",
            "Request For Payment SUBMITTED by EMPLOYEE"
        ],
        "support": 0.9171,
        "confidence": 0.8343,
        "category": "cross_path",
        "description": "'Permit FINAL_APPROVED by SUPERVISOR' and 'Request For Payment SUBMITTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c1030",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit APPROVED by ADMINISTRATION",
            "Request For Payment SUBMITTED by EMPLOYEE"
        ],
        "support": 0.8238,
        "confidence": 0.8164,
        "category": "cross_path",
        "description": "'Permit APPROVED by ADMINISTRATION' and 'Request For Payment SUBMITTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c1202",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Start trip",
            "Permit REJECTED by SUPERVISOR"
        ],
        "support": 1.0,
        "confidence": 0.981,
        "category": "cross_path",
        "description": "'Start trip' and 'Permit REJECTED by SUPERVISOR' cannot both occur in the same trace"
    },
    {
        "id": "c1241",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Permit APPROVED by SUPERVISOR",
            "Request For Payment SUBMITTED by EMPLOYEE"
        ],
        "support": 0.2508,
        "confidence": 0.8757,
        "category": "cross_path",
        "description": "'Permit APPROVED by SUPERVISOR' and 'Request For Payment SUBMITTED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c1276",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment REJECTED by ADMINISTRATION",
            "Permit APPROVED by BUDGET OWNER"
        ],
        "support": 0.3003,
        "confidence": 0.9691,
        "category": "cross_path",
        "description": "'Request For Payment REJECTED by ADMINISTRATION' and 'Permit APPROVED by BUDGET OWNER' cannot both occur in the same trace"
    },
    {
        "id": "c1313",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Declaration SUBMITTED by EMPLOYEE",
            "Send Reminder"
        ],
        "support": 0.9308,
        "confidence": 0.9399,
        "category": "cross_path",
        "description": "'Declaration SUBMITTED by EMPLOYEE' and 'Send Reminder' cannot both occur in the same trace"
    },
    {
        "id": "c1326",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Declaration SUBMITTED by EMPLOYEE",
            "Permit SAVED by EMPLOYEE"
        ],
        "support": 0.7935,
        "confidence": 1.0,
        "category": "cross_path",
        "description": "'Declaration SUBMITTED by EMPLOYEE' and 'Permit SAVED by EMPLOYEE' cannot both occur in the same trace"
    },
    {
        "id": "c1404",
        "template": "noncoexistence",
        "arity": "binary",
        "activities": [
            "Request For Payment APPROVED by BUDGET OWNER",
            "Declaration REJECTED by ADMINISTRATION"
        ],
        "support": 0.2141,
        "confidence": 0.9298,
        "category": "cross_path",
        "description": "'Request For Payment APPROVED by BUDGET OWNER' and 'Declaration REJECTED by ADMINISTRATION' cannot both occur in the same trace"
    }
]

# Ordered list of all unique activities referenced by the constraints above
UNIQUE_ACTIVITIES: list[str] = ['Permit SUBMITTED by EMPLOYEE', 'Declaration SUBMITTED by EMPLOYEE', 'End trip', 'Declaration APPROVED by ADMINISTRATION', 'Start trip', 'Declaration APPROVED by BUDGET OWNER', 'Request Payment', 'Payment Handled', 'Declaration FINAL_APPROVED by SUPERVISOR', 'Request For Payment SUBMITTED by EMPLOYEE', 'Request For Payment APPROVED by ADMINISTRATION', 'Declaration REJECTED by EMPLOYEE', 'Send Reminder', 'Permit FINAL_APPROVED by DIRECTOR', 'Declaration REJECTED by ADMINISTRATION', 'Request For Payment APPROVED by BUDGET OWNER', 'Permit APPROVED by BUDGET OWNER', 'Permit REJECTED by SUPERVISOR', 'Permit REJECTED by EMPLOYEE', 'Permit REJECTED by ADMINISTRATION', 'Request For Payment REJECTED by ADMINISTRATION', 'Permit FINAL_APPROVED by SUPERVISOR', 'Permit APPROVED by ADMINISTRATION', 'Permit APPROVED by SUPERVISOR', 'Permit SAVED by EMPLOYEE']


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
