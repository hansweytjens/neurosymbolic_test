"""
LTN-augmented transformer for outcome prediction.

Supports three integration levels (combinable via --level):
  feature      : per-trace LTN constraint satisfaction scores projected into input embeddings
  loss         : BCE loss + ltn_weight * constraint-violation consistency penalty (differentiable)
  intermediate : LTN features injected as a residual at the mid-point of the encoder stack

SatAgg (mean constraint satisfaction over the batch) is logged every epoch alongside
the task loss so training dynamics can be compared with baseline.py.

Usage:
  python -m predict.ltn --dataset bpi12 --level loss
  python -m predict.ltn --dataset bpi12 --level feature loss intermediate
"""

import importlib
import json
import os
import random
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

from data.predict.dataset import NeSyDataset, ModelConfig
from datasets.config import DATASET_REGISTRY

warnings.filterwarnings("ignore")


# ── Argument parsing ──────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description="LTN-augmented EventTransformer for outcome prediction"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset to train on",
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
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=lambda v: v if v == "random" else int(v), default="random")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides ModelConfig.learning_rate, default 0.001)")
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

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=model_dim, nhead=num_heads, batch_first=True, dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.mid_idx = max(0, num_layers // 2 - 1)

        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes),
        )
        self.sigmoid = nn.Sigmoid()

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

        numerical = torch.stack(numerical_list, dim=2)
        return torch.cat(embeddings_list + [numerical], dim=2)

    def forward(self, x: torch.Tensor, ltn_feats: torch.Tensor = None) -> torch.Tensor:
        h = self._get_embeddings(x)
        h = self.input_proj(h)
        h = h + self.positional_encoding[:, : h.size(1), :]

        if self.use_feature and ltn_feats is not None:
            ltn_exp = ltn_feats.unsqueeze(1).expand(-1, h.size(1), -1)
            h = h + self.ltn_input_proj(ltn_exp)

        for i, layer in enumerate(self.encoder_layers):
            h = layer(h)
            if self.use_intermediate and ltn_feats is not None and i == self.mid_idx:
                ltn_exp = ltn_feats.unsqueeze(1).expand(-1, h.size(1), -1)
                h = h + self.ltn_inter_proj(ltn_exp)

        pooled = h.mean(dim=1)
        return self.sigmoid(self.classifier(pooled))


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_activity_vocab(csv_path: str, read_kwargs: dict) -> dict:
    """
    Reconstruct the concept:name → integer-code mapping that the preprocessor uses.
    pd.Categorical assigns codes in sorted category order, starting at 1.
    """
    data = pd.read_csv(csv_path, **read_kwargs)
    cats = pd.Categorical(data["concept:name"]).categories
    return {cat: i + 1 for i, cat in enumerate(cats)}


def compute_batch_ltn(x, compute_fn, activity_vocab, activity_col_start, seq_len, device):
    ltn_feats = compute_fn(
        x, activity_vocab, activity_col_start=activity_col_start, seq_len=seq_len
    )
    sat_agg = ltn_feats.mean().item() if ltn_feats.numel() > 0 else 1.0
    return ltn_feats, sat_agg


# ── Main ──────────────────────────────────────────────────────────────────────

args = get_args()
if args.seed == "random":
    args.seed = random.randint(0, 2**32 - 1)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
print(f"-- Seed: {args.seed}")

cfg = DATASET_REGISTRY[args.dataset]

levels = set(args.level)
use_feature      = "feature"      in levels
use_loss         = "loss"         in levels
use_intermediate = "intermediate" in levels
use_ltn_feats    = use_feature or use_loss or use_intermediate

max_prefix_length = cfg["max_prefix_length"]
numerical_features = cfg["numerical_features"]
read_kwargs = cfg.get("read_kwargs", {})
seq_len = max_prefix_length

preprocessor = importlib.import_module(cfg["preprocessor"])

# Load LTN module if available
ltn_module = None
compute_level1_features = None
if use_ltn_feats:
    if cfg["ltn_module"] is None:
        raise ValueError(
            f"Dataset '{args.dataset}' has no LTN constraints module. "
            "Remove --level flags or add an ltn_module entry to the dataset registry."
        )
    ltn_module = importlib.import_module(cfg["ltn_module"])
    compute_level1_features = ltn_module.compute_level1_features

config = ModelConfig(
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    dropout_rate=args.dropout_rate,
    num_epochs=args.num_epochs,
    sequence_length=max_prefix_length,
    dataset=args.dataset,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"-- LTN integration level(s): {sorted(levels)}")
print(f"-- LTN weight (loss term):   {args.ltn_weight}")
print(f"-- Reading dataset: {args.dataset}")

data = pd.read_csv(f"data_processed/{args.dataset}.csv", **read_kwargs)

with open(f"data_processed/{args.dataset}_splits.json") as f:
    splits = json.load(f)
train_ids, val_ids, test_ids = splits["train_ids"], splits["val_ids"], splits["test_ids"]

(X_train, y_train, X_val, y_val, X_test, y_test, feature_names), vocab_sizes, _ = (
    preprocessor.preprocess_eventlog(
        data, args.seed, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids
    )
)

print("--- Label distribution")
print("--- Training set:", Counter(y_train))
print("--- Validation set:", Counter(y_val))
print("--- Test set:", Counter(y_test))
print(feature_names)

activity_vocab = build_activity_vocab(f"data_processed/{args.dataset}.csv", read_kwargs)
activity_col_start = feature_names.index("concept:name") * seq_len

# Determine N_ltn with a dummy forward pass
n_ltn = 0
if use_ltn_feats:
    with torch.no_grad():
        _dummy_x = torch.zeros(1, len(feature_names) * seq_len)
        _dummy_feats = compute_level1_features(
            _dummy_x, activity_vocab, activity_col_start=activity_col_start, seq_len=seq_len
        )
        n_ltn = _dummy_feats.size(1)
    print(f"-- concept:name column offset: {activity_col_start}")
    print(f"-- Active LTN constraints: {n_ltn}")

train_dataset = NeSyDataset(X_train, y_train)
train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataset   = NeSyDataset(X_val, y_val)
val_loader    = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_dataset  = NeSyDataset(X_test, y_test)
test_loader   = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

model = EventTransformerLTN(
    vocab_sizes,
    config,
    feature_names,
    numerical_features,
    model_dim=args.hidden_size,
    num_classes=1,
    max_len=max_prefix_length,
    num_layers=args.num_layers,
    dropout=args.dropout_rate,
    ltn_feature_dim=n_ltn if use_ltn_feats else 0,
    use_feature=use_feature,
    use_intermediate=use_intermediate,
).to(device)
print(f"-- Model parameters: {sum(p.numel() for p in model.parameters()):,}")
if ltn_module is not None:
    from collections import Counter as _Counter
    _tmpl_counts = _Counter(c["template"] for c in ltn_module.CONSTRAINTS)
    print(f"-- Constraints by template: { {k: _tmpl_counts[k] for k in sorted(_tmpl_counts)} }")

lr = args.lr if args.lr is not None else config.learning_rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
criterion = nn.BCELoss()
print(f"-- Learning rate: {lr}")

# ── Training loop ─────────────────────────────────────────────────────────────

os.makedirs("checkpoints", exist_ok=True)
level_str = "+".join(sorted(levels))
checkpoint_path = f"checkpoints/ltn_{args.dataset}_{level_str}_best.pt"

model.train()
training_losses   = []
validation_losses = []
best_val_loss     = float("inf")
patience_counter  = 0

for epoch in range(config.num_epochs):
    train_losses    = []
    train_sat_agg   = []
    train_ltn_contribs = []

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        ltn_feats = None
        if use_ltn_feats:
            ltn_feats, _ = compute_batch_ltn(
                x, compute_level1_features, activity_vocab, activity_col_start, seq_len, device
            )
            train_sat_agg.append((ltn_feats < 1.0).any(dim=1).float().mean().item())

        output = model(x, ltn_feats)
        bce_loss = criterion(output.squeeze(1), y)

        ltn_contrib = 0.0
        if use_ltn_feats and ltn_feats is not None and ltn_feats.numel() > 0:
            # binary: 1.0 for any violated constraint in the prefix, 0.0 for clean traces
            violation_scores = (ltn_feats < 1.0).any(dim=1).float()
            if use_loss:
                ltn_loss = (output.squeeze(1) * violation_scores).mean()
                ltn_contrib = (args.ltn_weight * ltn_loss).item()
                loss = bce_loss + args.ltn_weight * ltn_loss
            else:
                loss = bce_loss
        else:
            loss = bce_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())
        train_ltn_contribs.append(ltn_contrib)

    mean_train_loss = statistics.mean(train_losses)
    mean_viol_rate  = statistics.mean(train_sat_agg) if train_sat_agg else float("nan")
    mean_ltn_contrib = statistics.mean(train_ltn_contribs) if train_ltn_contribs else 0.0

    print(
        f"Epoch {epoch + 1}/{config.num_epochs} | "
        f"Loss: {mean_train_loss:.4f} | "
        f"ViolRate: {mean_viol_rate:.4f} | "
        f"LTNContrib: {mean_ltn_contrib:.5f}"
    )
    training_losses.append(mean_train_loss)

    model.eval()
    val_losses  = []
    val_sat_agg = []
    for x, y in val_loader:
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            ltn_feats = None
            if use_ltn_feats:
                ltn_feats, _ = compute_batch_ltn(
                    x, compute_level1_features, activity_vocab, activity_col_start, seq_len, device
                )
                val_sat_agg.append((ltn_feats < 1.0).any(dim=1).float().mean().item())
            output   = model(x, ltn_feats)
            val_loss = criterion(output.squeeze(1), y)
            val_losses.append(val_loss.item())

    mean_val_loss    = statistics.mean(val_losses)
    mean_val_viol    = statistics.mean(val_sat_agg) if val_sat_agg else float("nan")
    current_lr       = optimizer.param_groups[0]["lr"]
    print(
        f"           Val Loss: {mean_val_loss:.4f} | "
        f"ViolRate: {mean_val_viol:.4f} | "
        f"LR: {current_lr:.2e}"
    )
    validation_losses.append(mean_val_loss)
    scheduler.step(mean_val_loss)

    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), checkpoint_path)
    else:
        patience_counter += 1
        if patience_counter >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs")
            break

    model.train()

model.load_state_dict(torch.load(checkpoint_path))

# ── Evaluation ────────────────────────────────────────────────────────────────

model.eval()
y_pred = []
y_true = []
test_sat_agg = []

for x, y in test_loader:
    with torch.no_grad():
        x, y = x.to(device), y.to(device)

        ltn_feats = None
        if use_ltn_feats:
            ltn_feats, _ = compute_batch_ltn(
                x, compute_level1_features, activity_vocab, activity_col_start, seq_len, device
            )
            test_sat_agg.append((ltn_feats < 1.0).any(dim=1).float().mean().item())

        outputs     = model(x, ltn_feats)
        predictions = np.where(outputs.cpu().numpy() > 0.5, 1.0, 0.0).flatten()

        for i in range(len(y)):
            y_pred.append(predictions[i])
            y_true.append(y[i].cpu())

print(f"\n-- LTN Results ({args.dataset} | {level_str})")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred, average="macro"))
print("Precision:", precision_score(y_true, y_pred, average="macro"))
print("Recall:", recall_score(y_true, y_pred, average="macro"))
if test_sat_agg:
    print("ViolRate (test):", statistics.mean(test_sat_agg))
