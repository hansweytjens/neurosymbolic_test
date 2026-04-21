import json
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from model.transformer import EventTransformer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import statistics
from collections import Counter
from data.prepare import preprocess_bpi12
from data.predict.dataset import NeSyDataset, ModelConfig

import argparse

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

args = get_args()

dataset = "bpi12"
max_prefix_length = 40

config = ModelConfig(
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    dropout_rate=args.dropout_rate,
    num_epochs=args.num_epochs,
    sequence_length=max_prefix_length,
    dataset=dataset
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("-- Reading dataset")
data = pd.read_csv("data_processed/" + dataset + ".csv", dtype={"org:resource": str})

with open(f"data_processed/{dataset}_splits.json") as f:
    splits = json.load(f)
train_ids, val_ids, test_ids = splits["train_ids"], splits["val_ids"], splits["test_ids"]

(X_train, y_train, X_val, y_val, X_test, y_test, feature_names), vocab_sizes, scalers = preprocess_bpi12.preprocess_eventlog(
    data, args.seed, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids
)

print("--- Label distribution")
print("--- Training set:", Counter(y_train))
print("--- Validation set:", Counter(y_val))
print("--- Test set:", Counter(y_test))
print(feature_names)

numerical_features = ["case:AMOUNT_REQ", "elapsed_time", "time_since_previous"]

train_dataset = NeSyDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = NeSyDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataset = NeSyDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = EventTransformer(vocab_sizes, config, feature_names, numerical_features, model_dim=64, num_classes=1, max_len=max_prefix_length).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = torch.nn.BCELoss()

model.train()
training_losses = []
validation_losses = []
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
    print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {statistics.mean(train_losses)}")
    training_losses.append(statistics.mean(train_losses))

    model.eval()
    val_losses = []
    for x, y in val_loader:
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output.squeeze(1), y)
            val_losses.append(loss.item())
    print(f"Validation Loss: {statistics.mean(val_losses)}")
    validation_losses.append(statistics.mean(val_losses))
    if epoch >= 5 and validation_losses[-1] > validation_losses[-2]:
        print("Validation loss increased, stopping training")
        break
    model.train()

model.eval()
y_pred = []
y_true = []

rule_amount_1 = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1)
rule_amount_2 = lambda x: (x[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1)
rule_amount_3 = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1)
rule_resource_1 = lambda x: (x[:, :240] == 48).any(dim=1)
rule_resource_2 = lambda x: (x[:, :240] == 21).any(dim=1)

compliance = 0
num_constraints = 0
for x, y in test_loader:
    with torch.no_grad():
        x, y = x.to(device), y.to(device)
        r1 = rule_amount_1(x).cpu().numpy()
        r2 = rule_amount_2(x).cpu().numpy()
        r3 = rule_amount_3(x).cpu().numpy()
        r4 = rule_resource_1(x).cpu().numpy()
        r5 = rule_resource_2(x).cpu().numpy()
        outputs = model(x).cpu().numpy()
        predictions = np.where(outputs > 0.5, 1., 0.).flatten()
        for i in range(len(y)):
            y_pred.append(predictions[i])
            y_true.append(y[i].cpu())
            if r1[i] == 1 and y[i] == 0:
                num_constraints += 1
                if predictions[i] == 0:
                    compliance += 1
            if r2[i] == 1 and r3[i] == 1 and y[i] == 0:
                num_constraints += 1
                if predictions[i] == 0:
                    compliance += 1
            if r4[i] == 1 and y[i] == 0:
                num_constraints += 1
                if predictions[i] == 0:
                    compliance += 1
            if r5[i] == 1 and y[i] == 0:
                num_constraints += 1
                if predictions[i] == 0:
                    compliance += 1

print("\n-- Baseline Results")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred, average='macro'))
print("Precision:", precision_score(y_true, y_pred, average='macro'))
print("Recall:", recall_score(y_true, y_pred, average='macro'))
print("Compliance:", compliance / num_constraints if num_constraints > 0 else "N/A")
