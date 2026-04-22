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
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from model.transformer import EventTransformer
from data.predict.dataset import NeSyDataset, ModelConfig
from datasets.config import DATASET_REGISTRY

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Baseline EventTransformer for outcome prediction")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset to train on",
    )
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=lambda v: v if v == "random" else int(v), default=42)
    return parser.parse_args()


args = get_args()
if args.seed == "random":
    args.seed = random.randint(0, 2**32 - 1)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
print(f"-- Seed: {args.seed}")
cfg = DATASET_REGISTRY[args.dataset]

max_prefix_length = cfg["max_prefix_length"]
numerical_features = cfg["numerical_features"]
read_kwargs = cfg.get("read_kwargs", {})

preprocessor = importlib.import_module(cfg["preprocessor"])

config = ModelConfig(
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    dropout_rate=args.dropout_rate,
    num_epochs=args.num_epochs,
    sequence_length=max_prefix_length,
    dataset=args.dataset,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"-- Reading dataset: {args.dataset}")
data = pd.read_csv(f"data_processed/{args.dataset}.csv", **read_kwargs)

with open(f"data_processed/{args.dataset}_splits.json") as f:
    splits = json.load(f)
train_ids, val_ids, test_ids = splits["train_ids"], splits["val_ids"], splits["test_ids"]

(X_train, y_train, X_val, y_val, X_test, y_test, feature_names), vocab_sizes, scalers = (
    preprocessor.preprocess_eventlog(
        data, args.seed, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids
    )
)

print("--- Label distribution")
print("--- Training set:", Counter(y_train))
print("--- Validation set:", Counter(y_val))
print("--- Test set:", Counter(y_test))
print(feature_names)

train_dataset = NeSyDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = NeSyDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataset = NeSyDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = EventTransformer(
    vocab_sizes, config, feature_names, numerical_features,
    model_dim=64, num_classes=1, max_len=max_prefix_length,
).to(device)
print(f"-- Model parameters: {sum(p.numel() for p in model.parameters()):,}")
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = torch.nn.BCELoss()

os.makedirs("checkpoints", exist_ok=True)
checkpoint_path = f"checkpoints/baseline_{args.dataset}_best.pt"

model.train()
training_losses = []
validation_losses = []
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(config.num_epochs):
    train_losses = []
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.squeeze(1), y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {statistics.mean(train_losses):.4f}")
    training_losses.append(statistics.mean(train_losses))

    model.eval()
    val_losses = []
    for x, y in val_loader:
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output.squeeze(1), y)
            val_losses.append(loss.item())
    mean_val_loss = statistics.mean(val_losses)
    print(f"Validation Loss: {mean_val_loss:.4f}")
    validation_losses.append(mean_val_loss)

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
model.eval()
y_pred = []
y_true = []

for x, y in test_loader:
    with torch.no_grad():
        x, y = x.to(device), y.to(device)
        outputs = model(x).cpu().numpy()
        predictions = np.where(outputs > 0.5, 1.0, 0.0).flatten()
        y_pred.extend(predictions.tolist())
        y_true.extend(y.cpu().tolist())

print(f"\n-- Baseline Results ({args.dataset})")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred, average="macro"))
print("Precision:", precision_score(y_true, y_pred, average="macro"))
print("Recall:", recall_score(y_true, y_pred, average="macro"))
