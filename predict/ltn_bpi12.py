"""
LTN-augmented transformer for BPI12 outcome prediction.

Supports three integration levels (combinable via --level):
  feature      : per-trace LTN constraint satisfaction scores projected into input embeddings
  loss         : BCE loss + ltn_weight * constraint-violation consistency penalty (differentiable)
  intermediate : LTN features injected as a residual at the mid-point of the encoder stack

SatAgg (mean constraint satisfaction over the batch) is logged every epoch alongside
the task loss so training dynamics can be compared with baseline_bpi12.py.
"""

import json
import statistics
import argparse
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data.prepare import preprocess_bpi12
from data.predict.dataset import NeSyDataset, ModelConfig
from data.rules.bpi12_ltn_constraints import compute_level1_features

warnings.filterwarnings("ignore")


# ── Argument parsing ──────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description="LTN-augmented EventTransformer for BPI12 outcome prediction"
    )
    parser.add_argument(
        "--level",
        nargs="+",
        choices=["feature", "loss", "intermediate"],
        default=["loss"],
        help="LTN integration level(s); multiple values are allowed, e.g. --level feature loss",
    )
    parser.add_argument(
        "--ltn_weight",
        type=float,
        default=0.5,
        help="Weight for the LTN violation penalty (used with --level loss)",
    )
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── LTN-augmented transformer ─────────────────────────────────────────────────

class EventTransformerLTN(nn.Module):
    """
    Same architecture as EventTransformer (baseline) with three optional
    LTN-constraint injection points.

    ltn_feature_dim  : number of active LTN constraints; 0 disables feature/intermediate paths
    use_feature      : add projected LTN features as a residual on the input embeddings
    use_intermediate : add projected LTN features as a residual after the middle encoder layer
    """

    def __init__(
        self,
        vocab_sizes,
        config,
        feature_names,
        numerical_features,
        model_dim,
        num_classes,
        max_len,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        ltn_feature_dim=0,
        use_feature=False,
        use_intermediate=False,
    ):
        super().__init__()
        torch.manual_seed(42)
        self.config = config
        self.feature_names = feature_names
        self.numerical_features = numerical_features
        self.use_feature = use_feature and ltn_feature_dim > 0
        self.use_intermediate = use_intermediate and ltn_feature_dim > 0
        self.ltn_feature_dim = ltn_feature_dim

        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
            for feat, vocab_size in vocab_sizes.items()
        })

        transformer_input_size = (model_dim * len(self.embeddings)) + len(self.numerical_features)
        self.input_proj = nn.Linear(transformer_input_size, model_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, model_dim))

        if self.use_feature:
            self.ltn_input_proj = nn.Linear(ltn_feature_dim, model_dim)

        if self.use_intermediate:
            self.ltn_inter_proj = nn.Linear(ltn_feature_dim, model_dim)

        # Individual encoder layers so we can inject between them
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=model_dim, nhead=num_heads, batch_first=True, dropout=dropout
            )
            for _ in range(num_layers)
        ])
        # Inject after the last layer of the first half (0-indexed)
        self.mid_idx = max(0, num_layers // 2 - 1)

        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes),
        )
        self.sigmoid = nn.Sigmoid()

    # ------------------------------------------------------------------
    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = self.config.sequence_length
        embeddings_list = []
        numerical_list = []

        for name in self.embeddings.keys():
            idx = self.feature_names.index(name) * seq_len
            embeddings_list.append(self.embeddings[name](x[:, idx: idx + seq_len].long()))

        for name in self.numerical_features:
            idx = self.feature_names.index(name) * seq_len
            numerical_list.append(x[:, idx: idx + seq_len])

        numerical = torch.stack(numerical_list, dim=2)  # (B, seq_len, n_num)
        return torch.cat(embeddings_list + [numerical], dim=2)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        ltn_feats: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x         : (B, flat_features) — same layout as baseline
        ltn_feats : (B, N_ltn) — pre-computed constraint satisfaction scores; None when
                    neither feature nor intermediate injection is active
        """
        h = self._get_embeddings(x)           # (B, seq_len, emb_dim)
        h = self.input_proj(h)                # (B, seq_len, model_dim)
        h = h + self.positional_encoding[:, : h.size(1), :]

        # Feature-level injection: residual add before transformer
        if self.use_feature and ltn_feats is not None:
            ltn_exp = ltn_feats.unsqueeze(1).expand(-1, h.size(1), -1)  # (B, seq, N_ltn)
            h = h + self.ltn_input_proj(ltn_exp)

        # Run encoder layers, injecting LTN features at the midpoint if requested
        for i, layer in enumerate(self.encoder_layers):
            h = layer(h)
            if self.use_intermediate and ltn_feats is not None and i == self.mid_idx:
                ltn_exp = ltn_feats.unsqueeze(1).expand(-1, h.size(1), -1)
                h = h + self.ltn_inter_proj(ltn_exp)

        pooled = h.mean(dim=1)
        return self.sigmoid(self.classifier(pooled))


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_activity_vocab(raw_csv_path: str) -> dict:
    """
    Reconstruct the concept:name → integer-code mapping that preprocess_bpi12 uses.
    pd.Categorical assigns codes in sorted category order, starting at 1.
    """
    data = pd.read_csv(raw_csv_path, dtype={"org:resource": str})
    cats = pd.Categorical(data["concept:name"]).categories
    return {cat: i + 1 for i, cat in enumerate(cats)}


def compute_batch_ltn(x, activity_vocab, activity_col_start, seq_len, device):
    """
    Returns (ltn_feats, sat_agg) for one batch.
      ltn_feats : (B, N_active) — per-trace binary constraint satisfaction (float)
      sat_agg   : float scalar  — mean satisfaction across batch and constraints
    """
    ltn_feats = compute_level1_features(
        x, activity_vocab, activity_col_start=activity_col_start, seq_len=seq_len
    )
    sat_agg = ltn_feats.mean().item() if ltn_feats.numel() > 0 else 1.0
    return ltn_feats, sat_agg


# ── Main ──────────────────────────────────────────────────────────────────────

args = get_args()

levels = set(args.level)
use_feature      = "feature"      in levels
use_loss         = "loss"         in levels
use_intermediate = "intermediate" in levels
use_ltn_feats    = use_feature or use_loss or use_intermediate

dataset          = "bpi12"
max_prefix_length = 40
seq_len           = max_prefix_length

config = ModelConfig(
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    dropout_rate=args.dropout_rate,
    num_epochs=args.num_epochs,
    sequence_length=max_prefix_length,
    dataset=dataset,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"-- LTN integration level(s): {sorted(levels)}")
print(f"-- LTN weight (loss term):   {args.ltn_weight}")
print("-- Reading dataset")

data = pd.read_csv(f"data_processed/{dataset}.csv", dtype={"org:resource": str})

with open(f"data_processed/{dataset}_splits.json") as f:
    splits = json.load(f)
train_ids, val_ids, test_ids = splits["train_ids"], splits["val_ids"], splits["test_ids"]

(X_train, y_train, X_val, y_val, X_test, y_test, feature_names), vocab_sizes, scalers = (
    preprocess_bpi12.preprocess_eventlog(
        data, args.seed, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids
    )
)

print("--- Label distribution")
print("--- Training set:", Counter(y_train))
print("--- Validation set:", Counter(y_val))
print("--- Test set:", Counter(y_test))
print(feature_names)

numerical_features = ["case:AMOUNT_REQ", "elapsed_time", "time_since_previous"]

# Build activity vocab for LTN (same encoding as preprocess_bpi12)
activity_vocab = build_activity_vocab(f"data_processed/{dataset}.csv")

# concept:name column starts at its feature-name index * seq_len
activity_col_start = feature_names.index("concept:name") * seq_len

# Determine N_ltn (number of active constraints) with a dummy forward pass
with torch.no_grad():
    _flat_size = len(feature_names) * seq_len
    _dummy_x = torch.zeros(1, _flat_size)
    _dummy_feats = compute_level1_features(
        _dummy_x, activity_vocab, activity_col_start=activity_col_start, seq_len=seq_len
    )
    n_ltn = _dummy_feats.size(1)
print(f"-- concept:name column offset: {activity_col_start}")
print(f"-- Active LTN constraints: {n_ltn}")

train_dataset = NeSyDataset(X_train, y_train)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset   = NeSyDataset(X_val, y_val)
val_loader    = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataset  = NeSyDataset(X_test, y_test)
test_loader   = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = EventTransformerLTN(
    vocab_sizes,
    config,
    feature_names,
    numerical_features,
    model_dim=64,
    num_classes=1,
    max_len=max_prefix_length,
    ltn_feature_dim=n_ltn if use_ltn_feats else 0,
    use_feature=use_feature,
    use_intermediate=use_intermediate,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = nn.BCELoss()

# ── Training loop ─────────────────────────────────────────────────────────────

model.train()
training_losses    = []
validation_losses  = []

for epoch in range(config.num_epochs):
    train_losses  = []
    train_sat_agg = []

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        ltn_feats = None
        sat_agg_batch = 1.0
        if use_ltn_feats:
            ltn_feats, sat_agg_batch = compute_batch_ltn(x, activity_vocab, activity_col_start, seq_len, device)
            train_sat_agg.append(sat_agg_batch)

        output = model(x, ltn_feats)
        bce_loss = criterion(output.squeeze(1), y)

        if use_loss and ltn_feats is not None and ltn_feats.numel() > 0:
            # Per-trace violation score in [0, 1]: fraction of constraints violated
            violation_scores = 1.0 - ltn_feats.mean(dim=1)  # (B,)
            # Penalise confident positive predictions for constraint-violating traces.
            # This term is differentiable w.r.t. model parameters.
            ltn_loss = (output.squeeze(1) * violation_scores).mean()
            loss = bce_loss + args.ltn_weight * ltn_loss
        else:
            loss = bce_loss

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    mean_train_loss = statistics.mean(train_losses)
    mean_sat        = statistics.mean(train_sat_agg) if train_sat_agg else float("nan")

    print(
        f"Epoch {epoch + 1}/{config.num_epochs} | "
        f"Loss: {mean_train_loss:.4f} | "
        f"SatAgg: {mean_sat:.4f}"
    )
    training_losses.append(mean_train_loss)

    model.eval()
    val_losses    = []
    val_sat_agg   = []
    for x, y in val_loader:
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            ltn_feats = None
            if use_ltn_feats:
                ltn_feats, sat_batch = compute_batch_ltn(x, activity_vocab, activity_col_start, seq_len, device)
                val_sat_agg.append(sat_batch)
            output   = model(x, ltn_feats)
            val_loss = criterion(output.squeeze(1), y)
            val_losses.append(val_loss.item())

    mean_val_loss = statistics.mean(val_losses)
    mean_val_sat  = statistics.mean(val_sat_agg) if val_sat_agg else float("nan")
    print(
        f"           Val Loss: {mean_val_loss:.4f} | "
        f"Val SatAgg: {mean_val_sat:.4f}"
    )
    validation_losses.append(mean_val_loss)

    if epoch >= 5 and validation_losses[-1] > validation_losses[-2]:
        print("Validation loss increased, stopping training")
        break
    model.train()

# ── Evaluation ────────────────────────────────────────────────────────────────

model.eval()
y_pred = []
y_true = []

rule_amount_1  = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1)
rule_amount_2  = lambda x: (x[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1)
rule_amount_3  = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1)
rule_resource_1 = lambda x: (x[:, :240] == 48).any(dim=1)
rule_resource_2 = lambda x: (x[:, :240] == 21).any(dim=1)

compliance    = 0
num_constraints_checked = 0

for x, y in test_loader:
    with torch.no_grad():
        x, y = x.to(device), y.to(device)

        ltn_feats = None
        if use_ltn_feats:
            ltn_feats, _ = compute_batch_ltn(x, activity_vocab, activity_col_start, seq_len, device)

        r1 = rule_amount_1(x).cpu().numpy()
        r2 = rule_amount_2(x).cpu().numpy()
        r3 = rule_amount_3(x).cpu().numpy()
        r4 = rule_resource_1(x).cpu().numpy()
        r5 = rule_resource_2(x).cpu().numpy()

        outputs     = model(x, ltn_feats).cpu().numpy()
        predictions = np.where(outputs > 0.5, 1.0, 0.0).flatten()

        for i in range(len(y)):
            y_pred.append(predictions[i])
            y_true.append(y[i].cpu())
            if r1[i] == 1 and y[i] == 0:
                num_constraints_checked += 1
                if predictions[i] == 0:
                    compliance += 1
            if r2[i] == 1 and r3[i] == 1 and y[i] == 0:
                num_constraints_checked += 1
                if predictions[i] == 0:
                    compliance += 1
            if r4[i] == 1 and y[i] == 0:
                num_constraints_checked += 1
                if predictions[i] == 0:
                    compliance += 1
            if r5[i] == 1 and y[i] == 0:
                num_constraints_checked += 1
                if predictions[i] == 0:
                    compliance += 1

level_str = "+".join(sorted(levels))
print(f"\n-- LTN Results ({level_str})")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred, average="macro"))
print("Precision:", precision_score(y_true, y_pred, average="macro"))
print("Recall:", recall_score(y_true, y_pred, average="macro"))
print(
    "Compliance:",
    compliance / num_constraints_checked if num_constraints_checked > 0 else "N/A",
)
