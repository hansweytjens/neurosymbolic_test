# Two-Stage LTN w/ Rule Pruning

This repository contains the code for the paper "Neuro-Symbolic Learning for Predictive Process Monitoring via Two-Stage Logic Tensor Networks with Rule Pruning"

## Files

*   **`main_bpi12.py`**: Contains the code for the *BPIC2012* event log.
*   **`main_bpi17.py`**: Contains the code for the *BPIC2017* event log.
*   **`main_sepsis.py`**: Contains the code for the *SEPSIS* event log.
*   **`main_traffic.py`**: Contains the code for the *TRAFFIC FINES* dataset.
*   **`main_feat_eng.py`**: Contains the code for the LSTM-FE and TFR-FE models.
*   **`data/preprocess_bpi12.py`**: Contains the code for preprocessing the *BPIC2012* event log.
*   **`data/preprocess_bpi17.py`**: Contains the code for preprocessing the *BPIC2017* event log.
*   **`data/preprocess_sepsis.py`**: Contains the code for preprocessing the *Sepsis* event log.
*   **`data/preprocess_traffic.py`**: Contains the code for preprocessing the *TRAFFIC FINES* event log.
*   **`model/lstm.py`**: Contains the architecture used for the LSTM backbone.
*   **`model/transformer.py`**: Contains the architecture used
*   **`knowledge_base.py`**: Dataset class. for the Transformer backbone.
*   **`data/dataset.py`**: Dataset class.
*   **`knowledge_base.py`**: Dataset class.
*   **`create_temporal_features.py`**: Contains the code to create the temporal features also used for logical rules.

## Datasets

The tested event logs can be found at https://data.4tu.nl/search?datatypes=3

## Usage

Execute the script of interest with following flags:
* --backbone: "lstm" or "transformer"
* --seed: random seed used for parameters and splitting
* --num_epochs: number of training epochs of vanilla models
* --num_epochs_nesy: number of training epochs of LTN models
* --hidden_size: hidden_size of LSTM/Transformer backbones
* --num_layers: LSTM/Transformer layers
* --dropout_rate: dropout_rate for LSTM/Transformer backbones

Example for the *SEPSIS* event log with default parameters:

```python main_sepsis.py --model_type="lstm" --hidden_size=128 --num_layers=2 --seed=42```

---

## DECLARE Constraint Discovery and LTN Integration Pipeline

This pipeline discovers process constraints from training data, filters them to only those that carry a net-positive discriminative signal on the validation set, and feeds the survivors into the LTN model as input features and a violation penalty.

### Motivation

Not all discovered constraints are useful for LTN. A constraint penalises both wrong predictions that violate it (good) and correct predictions that violate it (bad). A constraint is net-positive only when it catches more wrong predictions than correct ones **in absolute counts** — a condition that depends on the class imbalance of the prediction set and must be evaluated on held-out data.

### Step 1 — Discover DECLARE constraints (training data only)

```bash
python rules/discover_declare.py --dataset bpi12
```

Mines constraints from the training split and writes `data/rules/bpi12_declare.json`. Configuration (per-template confidence floors, support threshold) lives in `DEFAULT_CONFIG` inside the script and can be overridden with a JSON file:

```bash
python rules/discover_declare.py --dataset bpi12 --config noncoex_095_override.json
```

### Step 2 — Score per-constraint discriminability (validation predictions)

```bash
python rules/check_declare_violations.py --dataset bpi12
```

Auto-detects the validation predictions file from `datasets/config.py` (`file_prefix` → `data/{prefix}_student_model_val_prefix_predictions.csv`). Pass `--predictions <path>` to override.

Use the **validation** set, never the test set — this step selects which constraints to include, making it a form of hyperparameter tuning. Using training predictions would introduce circularity (the constraints were mined from the same data).

Outputs:
- `data/{prefix}_student_model_val_declare_violation_analysis.csv` — per-row violation flags
- `data/{prefix}_student_model_val_declare_discriminability.csv` — per-constraint scores:

| column | meaning |
|---|---|
| `wrong_abs` | predictions the model got wrong that violated this constraint |
| `correct_abs` | predictions the model got right that violated this constraint |
| `net` | `wrong_abs − correct_abs` — positive means net benefit |
| `rate_ratio` | `(wrong_rate) / (correct_rate)` — useful for ranking, but `net` drives the filter |

A constraint is useful only when `net > 0`. Rate ratio alone is misleading when correct predictions far outnumber wrong ones (the typical case), because even a high ratio can produce a negative net.

### Step 3 — Generate the LTN constraints module

```bash
python rules/convert_declare_to_ltn.py --dataset bpi12 --min-net 1
```

Auto-detects the discriminability CSV produced in Step 2 (same `file_prefix` lookup). Pass `--discriminability-csv <path>` to override.

Reads the discriminability CSV, keeps every constraint with `net >= --min-net`, and writes `data/rules/bpi12_ltn_constraints.py`. The generated module exposes:

- `compute_level1_features(x, activity_vocab, activity_col_start, seq_len)` — returns a `[batch, N]` float tensor (1.0 = satisfied, 0.0 = violated) for all active constraints. Supports all discovered templates: `noncoexistence`, `precedence`, `altresponse`, `altprecedence`, `altsuccession`, `chain*`, and the existence-based family.
- `build_level2_formulas(...)` — LTN SatAgg formulas (existence-based templates only).
- `make_predicates(...)` / `make_constants(activity_vocab)` — Level 3 architecture hooks.

### Step 4 — Train the LTN model

```bash
python -m predict.ltn --dataset bpi12 --level feature loss
```

Integration levels (combinable):

| flag | effect |
|---|---|
| `feature` | constraint satisfaction scores projected as a residual into input embeddings |
| `loss` | BCE loss + `ltn_weight × mean(output × violation_score)` penalty |
| `intermediate` | constraint scores injected as a residual at the mid-point of the encoder |

Key flags: `--ltn_weight` (default 0.5), `--hidden_size`, `--num_layers`, `--num_epochs`, `--patience`, `--seed`.

### Full one-liner sequence

Works identically for any dataset — just change the `--dataset` value:

```bash
python rules/discover_declare.py --dataset bpi12
python rules/check_declare_violations.py --dataset bpi12
python rules/convert_declare_to_ltn.py --dataset bpi12 --min-net 1
python -m predict.ltn --dataset bpi12 --level feature loss
```

```bash
python rules/discover_declare.py --dataset bpi20prepaid
python rules/check_declare_violations.py --dataset bpi20prepaid
python rules/convert_declare_to_ltn.py --dataset bpi20prepaid --min-net 1
python -m predict.ltn --dataset bpi20prepaid --level feature loss
```