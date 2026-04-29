from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import random


def create_test_set(data, seed, ratio=0.2):
    random.seed(seed)
    unique_groups = list(data.groupby("case:concept:name").groups.keys())
    len_test_set = int(len(unique_groups) * ratio)
    test_ids = random.sample(unique_groups, len_test_set)
    training_ids = [x for x in unique_groups if x not in test_ids]
    return training_ids, test_ids


def create_ngrams(data, train_ids, val_ids, test_ids, window_size=20):
    ngrams_training, labels_training = [], []
    ngrams_val, labels_val = [], []
    ngrams_test, labels_test = [], []
    feature_names = []

    drop_cols = ["label", "case:concept:name", "time:timestamp"]

    for split_ids, ngrams_out, labels_out in [
        (train_ids, ngrams_training, labels_training),
        (val_ids,   ngrams_val,      labels_val),
        (test_ids,  ngrams_test,     labels_test),
    ]:
        split_data = data[data["case:concept:name"].isin(split_ids)]
        for _, group in split_data.groupby("case:concept:name"):
            group = group.reset_index(drop=True)
            label = int(group["label"].dropna().iloc[0])
            if len(group) > window_size:
                group = group.iloc[:window_size]
            group = group.drop(columns=drop_cols)
            feature_names = group.columns.tolist()
            for n in range(1, len(group)):
                labels_out.append(label)
                ngram_df = group.iloc[:n]
                cols = [list(col) for col in zip(*ngram_df.values.tolist())]
                cols = [c + [0] * (window_size - len(c)) for c in cols]
                ngrams_out.append(cols)

    print(feature_names)
    return ngrams_training, labels_training, ngrams_val, labels_val, ngrams_test, labels_test, feature_names


def preprocess_eventlog(data, seed=42, train_ids=None, val_ids=None, test_ids=None):
    vocab_sizes = {}
    scalers = {}

    data = data.dropna(subset=["case:concept:name"])
    print("Number of traces:", data["case:concept:name"].nunique())
    print("Columns:", data.columns.tolist())

    if train_ids is None or val_ids is None or test_ids is None:
        train_ids, test_ids = create_test_set(data, seed)
        data_train = data[data["case:concept:name"].isin(train_ids)]
        train_ids, val_ids = create_test_set(data_train, seed, ratio=0.2)

    print("Train:", len(train_ids), " Val:", len(val_ids), " Test:", len(test_ids))

    # Activity encoding
    data["concept:name"] = pd.Categorical(data["concept:name"])
    data["concept:name"] = data["concept:name"].cat.codes + 1
    vocab_sizes["concept:name"] = int(data["concept:name"].max())

    # Resource encoding
    if "org:resource" in data.columns:
        data["org:resource"] = data["org:resource"].fillna("UNKNOWN").astype(str)
        data["org:resource"] = pd.Categorical(data["org:resource"])
        data["org:resource"] = data["org:resource"].cat.codes + 1
        vocab_sizes["org:resource"] = int(data["org:resource"].max())

    # Numerical: requested amount (case-level, forward-filled)
    if "case:RequestedAmount" in data.columns:
        scaler_amount = MinMaxScaler()
        data["case:RequestedAmount"] = data["case:RequestedAmount"].ffill().fillna(0)
        data["case:RequestedAmount"] = scaler_amount.fit_transform(data[["case:RequestedAmount"]])
        scalers["case:RequestedAmount"] = scaler_amount

    scaler_elapsed = MinMaxScaler()
    data["elapsed_time"] = data["elapsed_time"].fillna(0)
    data["elapsed_time"] = scaler_elapsed.fit_transform(data[["elapsed_time"]])
    scalers["elapsed_time"] = scaler_elapsed

    scaler_time_prev = MinMaxScaler()
    data["time_since_previous"] = data["time_since_previous"].fillna(0)
    data["time_since_previous"] = scaler_time_prev.fit_transform(data[["time_since_previous"]])
    scalers["time_since_previous"] = scaler_time_prev

    # Keep only the columns needed for ngrams (drop extras not used as features)
    keep = ["case:concept:name", "concept:name", "time:timestamp", "label"]
    if "org:resource" in data.columns:
        keep.append("org:resource")
    if "case:RequestedAmount" in data.columns:
        keep.append("case:RequestedAmount")
    keep += ["elapsed_time", "time_since_previous"]
    data = data[[c for c in keep if c in data.columns]]

    return create_ngrams(data, train_ids, val_ids, test_ids), vocab_sizes, scalers
