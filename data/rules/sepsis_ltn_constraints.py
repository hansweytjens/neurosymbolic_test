"""
Auto-generated LTN constraints from DECLARE rules.

Dataset:   sepsis
Templates: ['coexistence', 'exactly_one', 'existence', 'init', 'responded_existence']
Total:     182 constraints  (coexistence=74, exactly_one=6, existence=9, init=1, responded_existence=92)

Usage in training script
------------------------
from data.rules.sepsis_ltn_constraints import (
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
        "id": "c0044",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "CRP",
            "Admission NC"
        ],
        "support": 0.9628,
        "confidence": 0.7187,
        "description": "If 'CRP' occurs, 'Admission NC' must also occur somewhere in the trace"
    },
    {
        "id": "c0045",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Return ER",
            "IV Antibiotics"
        ],
        "support": 0.2604,
        "confidence": 0.88,
        "description": "If 'Return ER' occurs, 'IV Antibiotics' must also occur somewhere in the trace"
    },
    {
        "id": "c0046",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "LacticAcid",
            "CRP"
        ],
        "support": 0.8125,
        "confidence": 1.0,
        "description": "If 'LacticAcid' occurs, 'CRP' must also occur somewhere in the trace"
    },
    {
        "id": "c0047",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "CRP",
            "IV Antibiotics"
        ],
        "support": 0.9628,
        "confidence": 0.8006,
        "description": "If 'CRP' occurs, 'IV Antibiotics' must also occur somewhere in the trace"
    },
    {
        "id": "c0048",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "IV Antibiotics",
            "IV Liquid"
        ],
        "support": 0.7708,
        "confidence": 0.917,
        "description": "If 'IV Antibiotics' occurs, 'IV Liquid' must also occur somewhere in the trace"
    },
    {
        "id": "c0049",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Admission NC",
            "CRP"
        ],
        "support": 0.7009,
        "confidence": 0.9873,
        "description": "If 'Admission NC' occurs, 'CRP' must also occur somewhere in the trace"
    },
    {
        "id": "c0050",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Release A",
            "CRP"
        ],
        "support": 0.5506,
        "confidence": 0.9838,
        "description": "If 'Release A' occurs, 'CRP' must also occur somewhere in the trace"
    },
    {
        "id": "c0051",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "IV Antibiotics",
            "ER Sepsis Triage"
        ],
        "support": 0.7708,
        "confidence": 1.0,
        "description": "If 'IV Antibiotics' occurs, 'ER Sepsis Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0052",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "IV Antibiotics",
            "Leucocytes"
        ],
        "support": 0.7708,
        "confidence": 1.0,
        "description": "If 'IV Antibiotics' occurs, 'Leucocytes' must also occur somewhere in the trace"
    },
    {
        "id": "c0053",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Admission NC",
            "ER Registration"
        ],
        "support": 0.7009,
        "confidence": 1.0,
        "description": "If 'Admission NC' occurs, 'ER Registration' must also occur somewhere in the trace"
    },
    {
        "id": "c0054",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Release A",
            "ER Registration"
        ],
        "support": 0.5506,
        "confidence": 1.0,
        "description": "If 'Release A' occurs, 'ER Registration' must also occur somewhere in the trace"
    },
    {
        "id": "c0055",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Release A",
            "IV Liquid"
        ],
        "support": 0.5506,
        "confidence": 0.7649,
        "description": "If 'Release A' occurs, 'IV Liquid' must also occur somewhere in the trace"
    },
    {
        "id": "c0056",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "IV Liquid",
            "ER Triage"
        ],
        "support": 0.7068,
        "confidence": 1.0,
        "description": "If 'IV Liquid' occurs, 'ER Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0057",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Registration",
            "IV Liquid"
        ],
        "support": 1.0,
        "confidence": 0.7068,
        "description": "If 'ER Registration' occurs, 'IV Liquid' must also occur somewhere in the trace"
    },
    {
        "id": "c0058",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Sepsis Triage",
            "ER Triage"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "If 'ER Sepsis Triage' occurs, 'ER Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0059",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "LacticAcid",
            "Admission NC"
        ],
        "support": 0.8125,
        "confidence": 0.7491,
        "description": "If 'LacticAcid' occurs, 'Admission NC' must also occur somewhere in the trace"
    },
    {
        "id": "c0060",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Leucocytes",
            "LacticAcid"
        ],
        "support": 0.9688,
        "confidence": 0.8387,
        "description": "If 'Leucocytes' occurs, 'LacticAcid' must also occur somewhere in the trace"
    },
    {
        "id": "c0061",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "LacticAcid",
            "IV Antibiotics"
        ],
        "support": 0.8125,
        "confidence": 0.9103,
        "description": "If 'LacticAcid' occurs, 'IV Antibiotics' must also occur somewhere in the trace"
    },
    {
        "id": "c0062",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Admission NC",
            "Leucocytes"
        ],
        "support": 0.7009,
        "confidence": 0.9936,
        "description": "If 'Admission NC' occurs, 'Leucocytes' must also occur somewhere in the trace"
    },
    {
        "id": "c0063",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Release A",
            "ER Sepsis Triage"
        ],
        "support": 0.5506,
        "confidence": 1.0,
        "description": "If 'Release A' occurs, 'ER Sepsis Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0064",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Release A",
            "Leucocytes"
        ],
        "support": 0.5506,
        "confidence": 0.9919,
        "description": "If 'Release A' occurs, 'Leucocytes' must also occur somewhere in the trace"
    },
    {
        "id": "c0065",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Registration",
            "ER Sepsis Triage"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "If 'ER Registration' occurs, 'ER Sepsis Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0066",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Registration",
            "Leucocytes"
        ],
        "support": 1.0,
        "confidence": 0.9688,
        "description": "If 'ER Registration' occurs, 'Leucocytes' must also occur somewhere in the trace"
    },
    {
        "id": "c0067",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "CRP",
            "IV Liquid"
        ],
        "support": 0.9628,
        "confidence": 0.7342,
        "description": "If 'CRP' occurs, 'IV Liquid' must also occur somewhere in the trace"
    },
    {
        "id": "c0068",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Admission NC",
            "IV Antibiotics"
        ],
        "support": 0.7009,
        "confidence": 0.8365,
        "description": "If 'Admission NC' occurs, 'IV Antibiotics' must also occur somewhere in the trace"
    },
    {
        "id": "c0069",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "IV Liquid",
            "CRP"
        ],
        "support": 0.7068,
        "confidence": 1.0,
        "description": "If 'IV Liquid' occurs, 'CRP' must also occur somewhere in the trace"
    },
    {
        "id": "c0070",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Sepsis Triage",
            "CRP"
        ],
        "support": 1.0,
        "confidence": 0.9628,
        "description": "If 'ER Sepsis Triage' occurs, 'CRP' must also occur somewhere in the trace"
    },
    {
        "id": "c0071",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Leucocytes",
            "ER Triage"
        ],
        "support": 0.9688,
        "confidence": 1.0,
        "description": "If 'Leucocytes' occurs, 'ER Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0072",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "IV Liquid",
            "ER Registration"
        ],
        "support": 0.7068,
        "confidence": 1.0,
        "description": "If 'IV Liquid' occurs, 'ER Registration' must also occur somewhere in the trace"
    },
    {
        "id": "c0073",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Return ER",
            "ER Sepsis Triage"
        ],
        "support": 0.2604,
        "confidence": 1.0,
        "description": "If 'Return ER' occurs, 'ER Sepsis Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0074",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Sepsis Triage",
            "ER Registration"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "If 'ER Sepsis Triage' occurs, 'ER Registration' must also occur somewhere in the trace"
    },
    {
        "id": "c0075",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "CRP",
            "ER Sepsis Triage"
        ],
        "support": 0.9628,
        "confidence": 1.0,
        "description": "If 'CRP' occurs, 'ER Sepsis Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0076",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Leucocytes",
            "CRP"
        ],
        "support": 0.9688,
        "confidence": 0.9923,
        "description": "If 'Leucocytes' occurs, 'CRP' must also occur somewhere in the trace"
    },
    {
        "id": "c0077",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "LacticAcid",
            "ER Registration"
        ],
        "support": 0.8125,
        "confidence": 1.0,
        "description": "If 'LacticAcid' occurs, 'ER Registration' must also occur somewhere in the trace"
    },
    {
        "id": "c0078",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "IV Liquid",
            "LacticAcid"
        ],
        "support": 0.7068,
        "confidence": 0.9684,
        "description": "If 'IV Liquid' occurs, 'LacticAcid' must also occur somewhere in the trace"
    },
    {
        "id": "c0079",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Sepsis Triage",
            "LacticAcid"
        ],
        "support": 1.0,
        "confidence": 0.8125,
        "description": "If 'ER Sepsis Triage' occurs, 'LacticAcid' must also occur somewhere in the trace"
    },
    {
        "id": "c0080",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "IV Liquid",
            "Leucocytes"
        ],
        "support": 0.7068,
        "confidence": 1.0,
        "description": "If 'IV Liquid' occurs, 'Leucocytes' must also occur somewhere in the trace"
    },
    {
        "id": "c0081",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Sepsis Triage",
            "Leucocytes"
        ],
        "support": 1.0,
        "confidence": 0.9688,
        "description": "If 'ER Sepsis Triage' occurs, 'Leucocytes' must also occur somewhere in the trace"
    },
    {
        "id": "c0082",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "IV Liquid",
            "Admission NC"
        ],
        "support": 0.7068,
        "confidence": 0.7621,
        "description": "If 'IV Liquid' occurs, 'Admission NC' must also occur somewhere in the trace"
    },
    {
        "id": "c0083",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Admission NC",
            "Release A"
        ],
        "support": 0.7009,
        "confidence": 0.7834,
        "description": "If 'Admission NC' occurs, 'Release A' must also occur somewhere in the trace"
    },
    {
        "id": "c0084",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Leucocytes",
            "ER Registration"
        ],
        "support": 0.9688,
        "confidence": 1.0,
        "description": "If 'Leucocytes' occurs, 'ER Registration' must also occur somewhere in the trace"
    },
    {
        "id": "c0085",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "LacticAcid",
            "IV Liquid"
        ],
        "support": 0.8125,
        "confidence": 0.8425,
        "description": "If 'LacticAcid' occurs, 'IV Liquid' must also occur somewhere in the trace"
    },
    {
        "id": "c0086",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Sepsis Triage",
            "Admission NC"
        ],
        "support": 1.0,
        "confidence": 0.7009,
        "description": "If 'ER Sepsis Triage' occurs, 'Admission NC' must also occur somewhere in the trace"
    },
    {
        "id": "c0087",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "IV Liquid",
            "IV Antibiotics"
        ],
        "support": 0.7068,
        "confidence": 1.0,
        "description": "If 'IV Liquid' occurs, 'IV Antibiotics' must also occur somewhere in the trace"
    },
    {
        "id": "c0088",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Sepsis Triage",
            "IV Antibiotics"
        ],
        "support": 1.0,
        "confidence": 0.7708,
        "description": "If 'ER Sepsis Triage' occurs, 'IV Antibiotics' must also occur somewhere in the trace"
    },
    {
        "id": "c0089",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Leucocytes",
            "IV Liquid"
        ],
        "support": 0.9688,
        "confidence": 0.7296,
        "description": "If 'Leucocytes' occurs, 'IV Liquid' must also occur somewhere in the trace"
    },
    {
        "id": "c0090",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Admission NC",
            "IV Liquid"
        ],
        "support": 0.7009,
        "confidence": 0.7686,
        "description": "If 'Admission NC' occurs, 'IV Liquid' must also occur somewhere in the trace"
    },
    {
        "id": "c0091",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "LacticAcid",
            "ER Sepsis Triage"
        ],
        "support": 0.8125,
        "confidence": 1.0,
        "description": "If 'LacticAcid' occurs, 'ER Sepsis Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0092",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "LacticAcid",
            "Leucocytes"
        ],
        "support": 0.8125,
        "confidence": 1.0,
        "description": "If 'LacticAcid' occurs, 'Leucocytes' must also occur somewhere in the trace"
    },
    {
        "id": "c0093",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Admission NC",
            "ER Sepsis Triage"
        ],
        "support": 0.7009,
        "confidence": 1.0,
        "description": "If 'Admission NC' occurs, 'ER Sepsis Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0094",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Leucocytes",
            "Admission NC"
        ],
        "support": 0.9688,
        "confidence": 0.7189,
        "description": "If 'Leucocytes' occurs, 'Admission NC' must also occur somewhere in the trace"
    },
    {
        "id": "c0095",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Leucocytes",
            "IV Antibiotics"
        ],
        "support": 0.9688,
        "confidence": 0.7957,
        "description": "If 'Leucocytes' occurs, 'IV Antibiotics' must also occur somewhere in the trace"
    },
    {
        "id": "c0096",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Triage",
            "LacticAcid"
        ],
        "support": 1.0,
        "confidence": 0.8125,
        "description": "If 'ER Triage' occurs, 'LacticAcid' must also occur somewhere in the trace"
    },
    {
        "id": "c0097",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Return ER",
            "LacticAcid"
        ],
        "support": 0.2604,
        "confidence": 0.9086,
        "description": "If 'Return ER' occurs, 'LacticAcid' must also occur somewhere in the trace"
    },
    {
        "id": "c0098",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "CRP",
            "LacticAcid"
        ],
        "support": 0.9628,
        "confidence": 0.8439,
        "description": "If 'CRP' occurs, 'LacticAcid' must also occur somewhere in the trace"
    },
    {
        "id": "c0099",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Triage",
            "Admission NC"
        ],
        "support": 1.0,
        "confidence": 0.7009,
        "description": "If 'ER Triage' occurs, 'Admission NC' must also occur somewhere in the trace"
    },
    {
        "id": "c0100",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Registration",
            "ER Triage"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "If 'ER Registration' occurs, 'ER Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0101",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Return ER",
            "Admission NC"
        ],
        "support": 0.2604,
        "confidence": 1.0,
        "description": "If 'Return ER' occurs, 'Admission NC' must also occur somewhere in the trace"
    },
    {
        "id": "c0102",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Triage",
            "IV Antibiotics"
        ],
        "support": 1.0,
        "confidence": 0.7708,
        "description": "If 'ER Triage' occurs, 'IV Antibiotics' must also occur somewhere in the trace"
    },
    {
        "id": "c0103",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Sepsis Triage",
            "IV Liquid"
        ],
        "support": 1.0,
        "confidence": 0.7068,
        "description": "If 'ER Sepsis Triage' occurs, 'IV Liquid' must also occur somewhere in the trace"
    },
    {
        "id": "c0104",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Return ER",
            "ER Triage"
        ],
        "support": 0.2604,
        "confidence": 1.0,
        "description": "If 'Return ER' occurs, 'ER Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0105",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "IV Liquid",
            "ER Sepsis Triage"
        ],
        "support": 0.7068,
        "confidence": 1.0,
        "description": "If 'IV Liquid' occurs, 'ER Sepsis Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0106",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "CRP",
            "ER Triage"
        ],
        "support": 0.9628,
        "confidence": 1.0,
        "description": "If 'CRP' occurs, 'ER Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0107",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Registration",
            "CRP"
        ],
        "support": 1.0,
        "confidence": 0.9628,
        "description": "If 'ER Registration' occurs, 'CRP' must also occur somewhere in the trace"
    },
    {
        "id": "c0108",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "IV Antibiotics",
            "LacticAcid"
        ],
        "support": 0.7708,
        "confidence": 0.9595,
        "description": "If 'IV Antibiotics' occurs, 'LacticAcid' must also occur somewhere in the trace"
    },
    {
        "id": "c0109",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "IV Antibiotics",
            "Admission NC"
        ],
        "support": 0.7708,
        "confidence": 0.7606,
        "description": "If 'IV Antibiotics' occurs, 'Admission NC' must also occur somewhere in the trace"
    },
    {
        "id": "c0110",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Triage",
            "CRP"
        ],
        "support": 1.0,
        "confidence": 0.9628,
        "description": "If 'ER Triage' occurs, 'CRP' must also occur somewhere in the trace"
    },
    {
        "id": "c0111",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Return ER",
            "CRP"
        ],
        "support": 0.2604,
        "confidence": 0.9943,
        "description": "If 'Return ER' occurs, 'CRP' must also occur somewhere in the trace"
    },
    {
        "id": "c0112",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "IV Antibiotics",
            "ER Triage"
        ],
        "support": 0.7708,
        "confidence": 1.0,
        "description": "If 'IV Antibiotics' occurs, 'ER Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0113",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Return ER",
            "Release A"
        ],
        "support": 0.2604,
        "confidence": 0.9029,
        "description": "If 'Return ER' occurs, 'Release A' must also occur somewhere in the trace"
    },
    {
        "id": "c0114",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Triage",
            "ER Registration"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "If 'ER Triage' occurs, 'ER Registration' must also occur somewhere in the trace"
    },
    {
        "id": "c0115",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Return ER",
            "ER Registration"
        ],
        "support": 0.2604,
        "confidence": 1.0,
        "description": "If 'Return ER' occurs, 'ER Registration' must also occur somewhere in the trace"
    },
    {
        "id": "c0116",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Leucocytes",
            "ER Sepsis Triage"
        ],
        "support": 0.9688,
        "confidence": 1.0,
        "description": "If 'Leucocytes' occurs, 'ER Sepsis Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0117",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "CRP",
            "ER Registration"
        ],
        "support": 0.9628,
        "confidence": 1.0,
        "description": "If 'CRP' occurs, 'ER Registration' must also occur somewhere in the trace"
    },
    {
        "id": "c0118",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Admission NC",
            "LacticAcid"
        ],
        "support": 0.7009,
        "confidence": 0.8684,
        "description": "If 'Admission NC' occurs, 'LacticAcid' must also occur somewhere in the trace"
    },
    {
        "id": "c0119",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Release A",
            "LacticAcid"
        ],
        "support": 0.5506,
        "confidence": 0.8486,
        "description": "If 'Release A' occurs, 'LacticAcid' must also occur somewhere in the trace"
    },
    {
        "id": "c0120",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Registration",
            "LacticAcid"
        ],
        "support": 1.0,
        "confidence": 0.8125,
        "description": "If 'ER Registration' occurs, 'LacticAcid' must also occur somewhere in the trace"
    },
    {
        "id": "c0121",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Triage",
            "IV Liquid"
        ],
        "support": 1.0,
        "confidence": 0.7068,
        "description": "If 'ER Triage' occurs, 'IV Liquid' must also occur somewhere in the trace"
    },
    {
        "id": "c0122",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Return ER",
            "IV Liquid"
        ],
        "support": 0.2604,
        "confidence": 0.8,
        "description": "If 'Return ER' occurs, 'IV Liquid' must also occur somewhere in the trace"
    },
    {
        "id": "c0123",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Release A",
            "Admission NC"
        ],
        "support": 0.5506,
        "confidence": 0.9973,
        "description": "If 'Release A' occurs, 'Admission NC' must also occur somewhere in the trace"
    },
    {
        "id": "c0124",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "LacticAcid",
            "ER Triage"
        ],
        "support": 0.8125,
        "confidence": 1.0,
        "description": "If 'LacticAcid' occurs, 'ER Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0125",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Registration",
            "Admission NC"
        ],
        "support": 1.0,
        "confidence": 0.7009,
        "description": "If 'ER Registration' occurs, 'Admission NC' must also occur somewhere in the trace"
    },
    {
        "id": "c0126",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Release A",
            "IV Antibiotics"
        ],
        "support": 0.5506,
        "confidence": 0.8351,
        "description": "If 'Release A' occurs, 'IV Antibiotics' must also occur somewhere in the trace"
    },
    {
        "id": "c0127",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "IV Antibiotics",
            "CRP"
        ],
        "support": 0.7708,
        "confidence": 1.0,
        "description": "If 'IV Antibiotics' occurs, 'CRP' must also occur somewhere in the trace"
    },
    {
        "id": "c0128",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Registration",
            "IV Antibiotics"
        ],
        "support": 1.0,
        "confidence": 0.7708,
        "description": "If 'ER Registration' occurs, 'IV Antibiotics' must also occur somewhere in the trace"
    },
    {
        "id": "c0129",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Triage",
            "ER Sepsis Triage"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "If 'ER Triage' occurs, 'ER Sepsis Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0130",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "ER Triage",
            "Leucocytes"
        ],
        "support": 1.0,
        "confidence": 0.9688,
        "description": "If 'ER Triage' occurs, 'Leucocytes' must also occur somewhere in the trace"
    },
    {
        "id": "c0131",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Admission NC",
            "ER Triage"
        ],
        "support": 0.7009,
        "confidence": 1.0,
        "description": "If 'Admission NC' occurs, 'ER Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0132",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Return ER",
            "Leucocytes"
        ],
        "support": 0.2604,
        "confidence": 0.9943,
        "description": "If 'Return ER' occurs, 'Leucocytes' must also occur somewhere in the trace"
    },
    {
        "id": "c0133",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "IV Antibiotics",
            "ER Registration"
        ],
        "support": 0.7708,
        "confidence": 1.0,
        "description": "If 'IV Antibiotics' occurs, 'ER Registration' must also occur somewhere in the trace"
    },
    {
        "id": "c0134",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "Release A",
            "ER Triage"
        ],
        "support": 0.5506,
        "confidence": 1.0,
        "description": "If 'Release A' occurs, 'ER Triage' must also occur somewhere in the trace"
    },
    {
        "id": "c0135",
        "template": "responded_existence",
        "arity": "binary",
        "activities": [
            "CRP",
            "Leucocytes"
        ],
        "support": 0.9628,
        "confidence": 0.9985,
        "description": "If 'CRP' occurs, 'Leucocytes' must also occur somewhere in the trace"
    },
    {
        "id": "c0256",
        "template": "exactly_one",
        "arity": "unary",
        "activities": [
            "ER Registration"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "'ER Registration' must occur exactly once"
    },
    {
        "id": "c0257",
        "template": "exactly_one",
        "arity": "unary",
        "activities": [
            "IV Antibiotics"
        ],
        "support": 1.0,
        "confidence": 0.7708,
        "description": "'IV Antibiotics' must occur exactly once"
    },
    {
        "id": "c0258",
        "template": "exactly_one",
        "arity": "unary",
        "activities": [
            "IV Liquid"
        ],
        "support": 1.0,
        "confidence": 0.7068,
        "description": "'IV Liquid' must occur exactly once"
    },
    {
        "id": "c0259",
        "template": "exactly_one",
        "arity": "unary",
        "activities": [
            "ER Sepsis Triage"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "'ER Sepsis Triage' must occur exactly once"
    },
    {
        "id": "c0260",
        "template": "exactly_one",
        "arity": "unary",
        "activities": [
            "LacticAcid"
        ],
        "support": 1.0,
        "confidence": 0.6771,
        "description": "'LacticAcid' must occur exactly once"
    },
    {
        "id": "c0261",
        "template": "exactly_one",
        "arity": "unary",
        "activities": [
            "ER Triage"
        ],
        "support": 1.0,
        "confidence": 0.997,
        "description": "'ER Triage' must occur exactly once"
    },
    {
        "id": "c0266",
        "template": "existence",
        "arity": "unary",
        "activities": [
            "LacticAcid"
        ],
        "support": 1.0,
        "confidence": 0.8125,
        "description": "'LacticAcid' must occur at least once"
    },
    {
        "id": "c0267",
        "template": "existence",
        "arity": "unary",
        "activities": [
            "ER Triage"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "'ER Triage' must occur at least once"
    },
    {
        "id": "c0268",
        "template": "existence",
        "arity": "unary",
        "activities": [
            "CRP"
        ],
        "support": 1.0,
        "confidence": 0.9628,
        "description": "'CRP' must occur at least once"
    },
    {
        "id": "c0269",
        "template": "existence",
        "arity": "unary",
        "activities": [
            "ER Registration"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "'ER Registration' must occur at least once"
    },
    {
        "id": "c0270",
        "template": "existence",
        "arity": "unary",
        "activities": [
            "Leucocytes"
        ],
        "support": 1.0,
        "confidence": 0.9688,
        "description": "'Leucocytes' must occur at least once"
    },
    {
        "id": "c0271",
        "template": "existence",
        "arity": "unary",
        "activities": [
            "Admission NC"
        ],
        "support": 1.0,
        "confidence": 0.7009,
        "description": "'Admission NC' must occur at least once"
    },
    {
        "id": "c0272",
        "template": "existence",
        "arity": "unary",
        "activities": [
            "IV Antibiotics"
        ],
        "support": 1.0,
        "confidence": 0.7708,
        "description": "'IV Antibiotics' must occur at least once"
    },
    {
        "id": "c0273",
        "template": "existence",
        "arity": "unary",
        "activities": [
            "IV Liquid"
        ],
        "support": 1.0,
        "confidence": 0.7068,
        "description": "'IV Liquid' must occur at least once"
    },
    {
        "id": "c0274",
        "template": "existence",
        "arity": "unary",
        "activities": [
            "ER Sepsis Triage"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "'ER Sepsis Triage' must occur at least once"
    },
    {
        "id": "c0278",
        "template": "init",
        "arity": "unary",
        "activities": [
            "ER Registration"
        ],
        "support": 1.0,
        "confidence": 0.942,
        "description": "'ER Registration' must be the first activity in every trace"
    },
    {
        "id": "c0327",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Sepsis Triage",
            "ER Registration"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "'ER Sepsis Triage' and 'ER Registration' must both occur, or neither occurs"
    },
    {
        "id": "c0328",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Sepsis Triage",
            "Leucocytes"
        ],
        "support": 1.0,
        "confidence": 0.9688,
        "description": "'ER Sepsis Triage' and 'Leucocytes' must both occur, or neither occurs"
    },
    {
        "id": "c0329",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Sepsis Triage",
            "LacticAcid"
        ],
        "support": 1.0,
        "confidence": 0.8125,
        "description": "'ER Sepsis Triage' and 'LacticAcid' must both occur, or neither occurs"
    },
    {
        "id": "c0330",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Sepsis Triage",
            "CRP"
        ],
        "support": 1.0,
        "confidence": 0.9628,
        "description": "'ER Sepsis Triage' and 'CRP' must both occur, or neither occurs"
    },
    {
        "id": "c0331",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Sepsis Triage",
            "IV Antibiotics"
        ],
        "support": 1.0,
        "confidence": 0.7708,
        "description": "'ER Sepsis Triage' and 'IV Antibiotics' must both occur, or neither occurs"
    },
    {
        "id": "c0332",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Sepsis Triage",
            "Admission NC"
        ],
        "support": 1.0,
        "confidence": 0.7009,
        "description": "'ER Sepsis Triage' and 'Admission NC' must both occur, or neither occurs"
    },
    {
        "id": "c0333",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Sepsis Triage",
            "ER Triage"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "'ER Sepsis Triage' and 'ER Triage' must both occur, or neither occurs"
    },
    {
        "id": "c0334",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Sepsis Triage",
            "IV Liquid"
        ],
        "support": 1.0,
        "confidence": 0.7068,
        "description": "'ER Sepsis Triage' and 'IV Liquid' must both occur, or neither occurs"
    },
    {
        "id": "c0335",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Registration",
            "ER Sepsis Triage"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "'ER Registration' and 'ER Sepsis Triage' must both occur, or neither occurs"
    },
    {
        "id": "c0336",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Registration",
            "Leucocytes"
        ],
        "support": 1.0,
        "confidence": 0.9688,
        "description": "'ER Registration' and 'Leucocytes' must both occur, or neither occurs"
    },
    {
        "id": "c0337",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Registration",
            "LacticAcid"
        ],
        "support": 1.0,
        "confidence": 0.8125,
        "description": "'ER Registration' and 'LacticAcid' must both occur, or neither occurs"
    },
    {
        "id": "c0338",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Registration",
            "CRP"
        ],
        "support": 1.0,
        "confidence": 0.9628,
        "description": "'ER Registration' and 'CRP' must both occur, or neither occurs"
    },
    {
        "id": "c0339",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Registration",
            "IV Antibiotics"
        ],
        "support": 1.0,
        "confidence": 0.7708,
        "description": "'ER Registration' and 'IV Antibiotics' must both occur, or neither occurs"
    },
    {
        "id": "c0340",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Registration",
            "Admission NC"
        ],
        "support": 1.0,
        "confidence": 0.7009,
        "description": "'ER Registration' and 'Admission NC' must both occur, or neither occurs"
    },
    {
        "id": "c0341",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Registration",
            "ER Triage"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "'ER Registration' and 'ER Triage' must both occur, or neither occurs"
    },
    {
        "id": "c0342",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Registration",
            "IV Liquid"
        ],
        "support": 1.0,
        "confidence": 0.7068,
        "description": "'ER Registration' and 'IV Liquid' must both occur, or neither occurs"
    },
    {
        "id": "c0343",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "Leucocytes",
            "ER Sepsis Triage"
        ],
        "support": 1.0,
        "confidence": 0.9688,
        "description": "'Leucocytes' and 'ER Sepsis Triage' must both occur, or neither occurs"
    },
    {
        "id": "c0344",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "Leucocytes",
            "ER Registration"
        ],
        "support": 1.0,
        "confidence": 0.9688,
        "description": "'Leucocytes' and 'ER Registration' must both occur, or neither occurs"
    },
    {
        "id": "c0345",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "Leucocytes",
            "LacticAcid"
        ],
        "support": 0.9688,
        "confidence": 0.8387,
        "description": "'Leucocytes' and 'LacticAcid' must both occur, or neither occurs"
    },
    {
        "id": "c0346",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "Leucocytes",
            "CRP"
        ],
        "support": 0.9702,
        "confidence": 0.9908,
        "description": "'Leucocytes' and 'CRP' must both occur, or neither occurs"
    },
    {
        "id": "c0347",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "Leucocytes",
            "IV Antibiotics"
        ],
        "support": 0.9688,
        "confidence": 0.7957,
        "description": "'Leucocytes' and 'IV Antibiotics' must both occur, or neither occurs"
    },
    {
        "id": "c0348",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "Leucocytes",
            "Admission NC"
        ],
        "support": 0.9732,
        "confidence": 0.7156,
        "description": "'Leucocytes' and 'Admission NC' must both occur, or neither occurs"
    },
    {
        "id": "c0349",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "Leucocytes",
            "ER Triage"
        ],
        "support": 1.0,
        "confidence": 0.9688,
        "description": "'Leucocytes' and 'ER Triage' must both occur, or neither occurs"
    },
    {
        "id": "c0350",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "Leucocytes",
            "IV Liquid"
        ],
        "support": 0.9688,
        "confidence": 0.7296,
        "description": "'Leucocytes' and 'IV Liquid' must both occur, or neither occurs"
    },
    {
        "id": "c0351",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "LacticAcid",
            "ER Sepsis Triage"
        ],
        "support": 1.0,
        "confidence": 0.8125,
        "description": "'LacticAcid' and 'ER Sepsis Triage' must both occur, or neither occurs"
    },
    {
        "id": "c0352",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "LacticAcid",
            "ER Registration"
        ],
        "support": 1.0,
        "confidence": 0.8125,
        "description": "'LacticAcid' and 'ER Registration' must both occur, or neither occurs"
    },
    {
        "id": "c0353",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "LacticAcid",
            "Leucocytes"
        ],
        "support": 0.9688,
        "confidence": 0.8387,
        "description": "'LacticAcid' and 'Leucocytes' must both occur, or neither occurs"
    },
    {
        "id": "c0354",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "LacticAcid",
            "CRP"
        ],
        "support": 0.9628,
        "confidence": 0.8439,
        "description": "'LacticAcid' and 'CRP' must both occur, or neither occurs"
    },
    {
        "id": "c0355",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "LacticAcid",
            "IV Antibiotics"
        ],
        "support": 0.8438,
        "confidence": 0.8765,
        "description": "'LacticAcid' and 'IV Antibiotics' must both occur, or neither occurs"
    },
    {
        "id": "c0356",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "LacticAcid",
            "Admission NC"
        ],
        "support": 0.9048,
        "confidence": 0.6727,
        "description": "'LacticAcid' and 'Admission NC' must both occur, or neither occurs"
    },
    {
        "id": "c0357",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "LacticAcid",
            "ER Triage"
        ],
        "support": 1.0,
        "confidence": 0.8125,
        "description": "'LacticAcid' and 'ER Triage' must both occur, or neither occurs"
    },
    {
        "id": "c0358",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "LacticAcid",
            "IV Liquid"
        ],
        "support": 0.8348,
        "confidence": 0.82,
        "description": "'LacticAcid' and 'IV Liquid' must both occur, or neither occurs"
    },
    {
        "id": "c0359",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "CRP",
            "ER Sepsis Triage"
        ],
        "support": 1.0,
        "confidence": 0.9628,
        "description": "'CRP' and 'ER Sepsis Triage' must both occur, or neither occurs"
    },
    {
        "id": "c0360",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "CRP",
            "ER Registration"
        ],
        "support": 1.0,
        "confidence": 0.9628,
        "description": "'CRP' and 'ER Registration' must both occur, or neither occurs"
    },
    {
        "id": "c0361",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "CRP",
            "Leucocytes"
        ],
        "support": 0.9702,
        "confidence": 0.9908,
        "description": "'CRP' and 'Leucocytes' must both occur, or neither occurs"
    },
    {
        "id": "c0362",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "CRP",
            "LacticAcid"
        ],
        "support": 0.9628,
        "confidence": 0.8439,
        "description": "'CRP' and 'LacticAcid' must both occur, or neither occurs"
    },
    {
        "id": "c0363",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "CRP",
            "IV Antibiotics"
        ],
        "support": 0.9628,
        "confidence": 0.8006,
        "description": "'CRP' and 'IV Antibiotics' must both occur, or neither occurs"
    },
    {
        "id": "c0364",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "CRP",
            "Admission NC"
        ],
        "support": 0.9717,
        "confidence": 0.7121,
        "description": "'CRP' and 'Admission NC' must both occur, or neither occurs"
    },
    {
        "id": "c0365",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "CRP",
            "ER Triage"
        ],
        "support": 1.0,
        "confidence": 0.9628,
        "description": "'CRP' and 'ER Triage' must both occur, or neither occurs"
    },
    {
        "id": "c0366",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "CRP",
            "IV Liquid"
        ],
        "support": 0.9628,
        "confidence": 0.7342,
        "description": "'CRP' and 'IV Liquid' must both occur, or neither occurs"
    },
    {
        "id": "c0367",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "IV Antibiotics",
            "ER Sepsis Triage"
        ],
        "support": 1.0,
        "confidence": 0.7708,
        "description": "'IV Antibiotics' and 'ER Sepsis Triage' must both occur, or neither occurs"
    },
    {
        "id": "c0368",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "IV Antibiotics",
            "ER Registration"
        ],
        "support": 1.0,
        "confidence": 0.7708,
        "description": "'IV Antibiotics' and 'ER Registration' must both occur, or neither occurs"
    },
    {
        "id": "c0369",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "IV Antibiotics",
            "Leucocytes"
        ],
        "support": 0.9688,
        "confidence": 0.7957,
        "description": "'IV Antibiotics' and 'Leucocytes' must both occur, or neither occurs"
    },
    {
        "id": "c0370",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "IV Antibiotics",
            "LacticAcid"
        ],
        "support": 0.8438,
        "confidence": 0.8765,
        "description": "'IV Antibiotics' and 'LacticAcid' must both occur, or neither occurs"
    },
    {
        "id": "c0371",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "IV Antibiotics",
            "CRP"
        ],
        "support": 0.9628,
        "confidence": 0.8006,
        "description": "'IV Antibiotics' and 'CRP' must both occur, or neither occurs"
    },
    {
        "id": "c0372",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "IV Antibiotics",
            "Admission NC"
        ],
        "support": 0.8854,
        "confidence": 0.6622,
        "description": "'IV Antibiotics' and 'Admission NC' must both occur, or neither occurs"
    },
    {
        "id": "c0373",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "IV Antibiotics",
            "ER Triage"
        ],
        "support": 1.0,
        "confidence": 0.7708,
        "description": "'IV Antibiotics' and 'ER Triage' must both occur, or neither occurs"
    },
    {
        "id": "c0374",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "IV Antibiotics",
            "IV Liquid"
        ],
        "support": 0.7708,
        "confidence": 0.917,
        "description": "'IV Antibiotics' and 'IV Liquid' must both occur, or neither occurs"
    },
    {
        "id": "c0375",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "Admission NC",
            "ER Sepsis Triage"
        ],
        "support": 1.0,
        "confidence": 0.7009,
        "description": "'Admission NC' and 'ER Sepsis Triage' must both occur, or neither occurs"
    },
    {
        "id": "c0376",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "Admission NC",
            "ER Registration"
        ],
        "support": 1.0,
        "confidence": 0.7009,
        "description": "'Admission NC' and 'ER Registration' must both occur, or neither occurs"
    },
    {
        "id": "c0377",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "Admission NC",
            "Leucocytes"
        ],
        "support": 0.9732,
        "confidence": 0.7156,
        "description": "'Admission NC' and 'Leucocytes' must both occur, or neither occurs"
    },
    {
        "id": "c0378",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "Admission NC",
            "LacticAcid"
        ],
        "support": 0.9048,
        "confidence": 0.6727,
        "description": "'Admission NC' and 'LacticAcid' must both occur, or neither occurs"
    },
    {
        "id": "c0379",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "Admission NC",
            "CRP"
        ],
        "support": 0.9717,
        "confidence": 0.7121,
        "description": "'Admission NC' and 'CRP' must both occur, or neither occurs"
    },
    {
        "id": "c0380",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "Admission NC",
            "IV Antibiotics"
        ],
        "support": 0.8854,
        "confidence": 0.6622,
        "description": "'Admission NC' and 'IV Antibiotics' must both occur, or neither occurs"
    },
    {
        "id": "c0381",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "Admission NC",
            "ER Triage"
        ],
        "support": 1.0,
        "confidence": 0.7009,
        "description": "'Admission NC' and 'ER Triage' must both occur, or neither occurs"
    },
    {
        "id": "c0382",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "Admission NC",
            "IV Liquid"
        ],
        "support": 0.869,
        "confidence": 0.6199,
        "description": "'Admission NC' and 'IV Liquid' must both occur, or neither occurs"
    },
    {
        "id": "c0383",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "Admission NC",
            "Release A"
        ],
        "support": 0.7024,
        "confidence": 0.7818,
        "description": "'Admission NC' and 'Release A' must both occur, or neither occurs"
    },
    {
        "id": "c0384",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Triage",
            "ER Sepsis Triage"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "'ER Triage' and 'ER Sepsis Triage' must both occur, or neither occurs"
    },
    {
        "id": "c0385",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Triage",
            "ER Registration"
        ],
        "support": 1.0,
        "confidence": 1.0,
        "description": "'ER Triage' and 'ER Registration' must both occur, or neither occurs"
    },
    {
        "id": "c0386",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Triage",
            "Leucocytes"
        ],
        "support": 1.0,
        "confidence": 0.9688,
        "description": "'ER Triage' and 'Leucocytes' must both occur, or neither occurs"
    },
    {
        "id": "c0387",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Triage",
            "LacticAcid"
        ],
        "support": 1.0,
        "confidence": 0.8125,
        "description": "'ER Triage' and 'LacticAcid' must both occur, or neither occurs"
    },
    {
        "id": "c0388",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Triage",
            "CRP"
        ],
        "support": 1.0,
        "confidence": 0.9628,
        "description": "'ER Triage' and 'CRP' must both occur, or neither occurs"
    },
    {
        "id": "c0389",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Triage",
            "IV Antibiotics"
        ],
        "support": 1.0,
        "confidence": 0.7708,
        "description": "'ER Triage' and 'IV Antibiotics' must both occur, or neither occurs"
    },
    {
        "id": "c0390",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Triage",
            "Admission NC"
        ],
        "support": 1.0,
        "confidence": 0.7009,
        "description": "'ER Triage' and 'Admission NC' must both occur, or neither occurs"
    },
    {
        "id": "c0391",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "ER Triage",
            "IV Liquid"
        ],
        "support": 1.0,
        "confidence": 0.7068,
        "description": "'ER Triage' and 'IV Liquid' must both occur, or neither occurs"
    },
    {
        "id": "c0392",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "IV Liquid",
            "ER Sepsis Triage"
        ],
        "support": 1.0,
        "confidence": 0.7068,
        "description": "'IV Liquid' and 'ER Sepsis Triage' must both occur, or neither occurs"
    },
    {
        "id": "c0393",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "IV Liquid",
            "ER Registration"
        ],
        "support": 1.0,
        "confidence": 0.7068,
        "description": "'IV Liquid' and 'ER Registration' must both occur, or neither occurs"
    },
    {
        "id": "c0394",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "IV Liquid",
            "Leucocytes"
        ],
        "support": 0.9688,
        "confidence": 0.7296,
        "description": "'IV Liquid' and 'Leucocytes' must both occur, or neither occurs"
    },
    {
        "id": "c0395",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "IV Liquid",
            "LacticAcid"
        ],
        "support": 0.8348,
        "confidence": 0.82,
        "description": "'IV Liquid' and 'LacticAcid' must both occur, or neither occurs"
    },
    {
        "id": "c0396",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "IV Liquid",
            "CRP"
        ],
        "support": 0.9628,
        "confidence": 0.7342,
        "description": "'IV Liquid' and 'CRP' must both occur, or neither occurs"
    },
    {
        "id": "c0397",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "IV Liquid",
            "IV Antibiotics"
        ],
        "support": 0.7708,
        "confidence": 0.917,
        "description": "'IV Liquid' and 'IV Antibiotics' must both occur, or neither occurs"
    },
    {
        "id": "c0398",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "IV Liquid",
            "Admission NC"
        ],
        "support": 0.869,
        "confidence": 0.6199,
        "description": "'IV Liquid' and 'Admission NC' must both occur, or neither occurs"
    },
    {
        "id": "c0399",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "IV Liquid",
            "ER Triage"
        ],
        "support": 1.0,
        "confidence": 0.7068,
        "description": "'IV Liquid' and 'ER Triage' must both occur, or neither occurs"
    },
    {
        "id": "c0400",
        "template": "coexistence",
        "arity": "binary",
        "activities": [
            "Release A",
            "Admission NC"
        ],
        "support": 0.7024,
        "confidence": 0.7818,
        "description": "'Release A' and 'Admission NC' must both occur, or neither occurs"
    }
]

# Ordered list of all unique activities referenced by the constraints above
UNIQUE_ACTIVITIES: list[str] = ['CRP', 'Admission NC', 'Return ER', 'IV Antibiotics', 'LacticAcid', 'IV Liquid', 'Release A', 'ER Sepsis Triage', 'Leucocytes', 'ER Registration', 'ER Triage']


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
                      in preprocess_sepsis.py)

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
