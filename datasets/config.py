"""
Central dataset registry.

Each entry contains everything that varies between datasets:
  preprocessor       - dotted module path with a preprocess_eventlog() function
  max_prefix_length  - sequence window size used in create_ngrams
  numerical_features - columns that are scaled (not embedded)
  read_kwargs        - extra kwargs forwarded to pd.read_csv for the processed CSV
  xes_file           - default path to the raw XES (or XES.gz) file
  ltn_module         - dotted module path with compute_level1_features(), or None
"""

DATASET_REGISTRY = {
    "bpi12": {
        "preprocessor": "data.prepare.preprocess_bpi12",
        "max_prefix_length": 40,
        "numerical_features": ["case:AMOUNT_REQ", "elapsed_time", "time_since_previous"],
        "read_kwargs": {"dtype": {"org:resource": str}},
        "xes_file": "bpi12_extracted/BPI_Challenge_2012.xes.gz",
        "ltn_module": "data.rules.bpi12_ltn_constraints",
    },
    "bpi17": {
        "preprocessor": "data.prepare.preprocess_bpi17",
        "max_prefix_length": 20,
        "numerical_features": [
            "case:RequestedAmount", "FirstWithdrawalAmount", "MonthlyCost",
            "CreditScore", "OfferedAmount", "elapsed_time", "time_since_previous",
        ],
        "read_kwargs": {},
        "xes_file": "data/bpi17.xes.gz",
        "ltn_module": None,
    },
    "sepsis": {
        "preprocessor": "data.prepare.preprocess_sepsis",
        "max_prefix_length": 13,
        "numerical_features": [
            "LacticAcid", "CRP", "Leucocytes", "Age",
            "elapsed_time", "time_since_previous",
        ],
        "read_kwargs": {},
        "xes_file": "data/sepsis.xes.gz",
        "ltn_module": "data.rules.sepsis_ltn_constraints",
    },
    "traffic": {
        "preprocessor": "data.prepare.preprocess_traffic",
        "max_prefix_length": 10,
        "numerical_features": [
            "amount", "paymentAmount", "expense",
            "elapsed_time", "time_since_previous",
        ],
        "read_kwargs": {},
        "xes_file": "data/traffic.xes.gz",
        "ltn_module": None,
    },
}
