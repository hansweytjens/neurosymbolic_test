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
        "file_prefix": "BPI12",
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
        "file_prefix": "BPI17",
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
        "file_prefix": "sepsis",
    },
    "bpi20permit": {
        "preprocessor": "data.prepare.preprocess_bpi20permit",
        "max_prefix_length": 30,
        "numerical_features": ["case:RequestedBudget", "case:TotalDeclared", "elapsed_time", "time_since_previous"],
        "read_kwargs": {"dtype": {"org:resource": str}},
        "xes_file": "data/bpi20permit.xes.gz",
        "ltn_module": "data.rules.bpi20permit_ltn_constraints",
        "file_prefix": "BPI20TravelPermitData",
    },
    "bpi20prepaid": {
        "preprocessor": "data.prepare.preprocess_bpi20prepaid",
        "max_prefix_length": 20,
        "numerical_features": ["case:RequestedAmount", "elapsed_time", "time_since_previous"],
        "read_kwargs": {"dtype": {"org:resource": str}},
        "xes_file": "data/bpi20prepaid.xes.gz",
        "ltn_module": "data.rules.bpi20prepaid_ltn_constraints",
        "file_prefix": "BPI20PrepaidTravelCosts",
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
        "file_prefix": "traffic",
    },
}


def get_file_prefix(dataset: str) -> str:
    """Return the file-naming prefix for a dataset (used to locate prediction and analysis files)."""
    return DATASET_REGISTRY[dataset].get("file_prefix", dataset.upper())
