from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import random

def create_test_set(data, seed, ratio=0.2):
    random.seed(seed)
    unique_groups = list(data.groupby("case:concept:name").groups.keys())
    len_test_set = int(len(unique_groups) * ratio)
    test_ids = random.sample(unique_groups, len_test_set)
    training_ids = [x for x in unique_groups if x not in test_ids]
    return training_ids, test_ids

def create_ngrams(data, train_ids, val_ids, test_ids, window_size=40):

    ngrams_test = []
    ngrams_training = []
    labels_training = []
    labels_test = []
    ngrams_val = []
    labels_val = []

    training_data = data[data["case:concept:name"].isin(train_ids)]
    val_data = data[data["case:concept:name"].isin(val_ids)]
    test_data = data[data["case:concept:name"].isin(test_ids)]

    for _, group in training_data.groupby('case:concept:name'):

        group = group.reset_index(drop=True)
        
        label = int(group['label'].dropna().iloc[0])

        if len(group) > window_size:
            group = group.iloc[:window_size]

        group = group.drop(columns=["label", "case:concept:name", "time:timestamp", "concept:name_str"])

        feature_names = group.columns.tolist()
        for n in range(1, len(group), 1):
            labels_training.append(label)
            ngram_df = group.iloc[:n]
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (window_size-len(inner_list)) for inner_list in cols]
            ngrams_training.append(cols)

    for _, group in val_data.groupby('case:concept:name'):

        group = group.reset_index(drop=True)
                
        label = int(group['label'].dropna().iloc[0])
        if len(group) > window_size:
            group = group.iloc[:window_size]

        group = group.drop(columns=["label", "case:concept:name", "time:timestamp", "concept:name_str"])
        
        feature_names = group.columns.tolist()
        for n in range(1, len(group), 1):
            labels_val.append(label)
            ngram_df = group.iloc[:n]
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (window_size-len(inner_list)) for inner_list in cols]
            ngrams_val.append(cols)

    for _, group in test_data.groupby('case:concept:name'):

        group = group.reset_index(drop=True)
                
        label = int(group['label'].dropna().iloc[0])
        if len(group) > window_size:
            group = group.iloc[:window_size]

        group = group.drop(columns=["label", "case:concept:name", "time:timestamp", "concept:name_str"])
        
        feature_names = group.columns.tolist()
        for n in range(1, len(group), 1):
            labels_test.append(label)
            ngram_df = group.iloc[:n]
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (window_size-len(inner_list)) for inner_list in cols]
            ngrams_test.append(cols)

    print(feature_names)

    return ngrams_training, labels_training, ngrams_val, labels_val, ngrams_test, labels_test, feature_names

def preprocess_eventlog(data, seed, train_ids=None, val_ids=None, test_ids=None, dataset_size=None):

    vocab_sizes = {}
    cases = data[data["concept:name"] == "A_SUBMITTED-COMPLETE"]
    labels = cases["label"].to_list()
    case_ids = cases["case:concept:name"].to_list()
    print("Number of traces: ", len(labels))

    if train_ids is None or val_ids is None or test_ids is None:
        train_ids, test_ids = create_test_set(data, seed)
        data_training = data[data["case:concept:name"].isin(train_ids)]
        train_ids, val_ids = create_test_set(data_training, seed, ratio=0.2)

    print("Number of traces in train set: ", len(train_ids))
    print("Number of traces in val set: ", len(val_ids))
    print("Number of traces in test set: ", len(test_ids))

    scaler_ar = MinMaxScaler()
    scaler_elapsed = MinMaxScaler()
    scaler_time_prev = MinMaxScaler()
    
    labels = data.groupby("case:concept:name")["label"].first().reset_index()
    print(data.columns)
    data = data.drop(columns=["case:REG_DATE"])

    data["concept:name_str"] = data["concept:name"]
    print("concept:name_str: ", data["concept:name_str"].unique())
    data["concept:name"] = pd.Categorical(data["concept:name"])
    print("W_Completeren aanvraag-COMPLETE: ", data["concept:name"].cat.categories.get_loc("W_Completeren aanvraag-COMPLETE") + 1)
    print("W_Valideren aanvraag-COMPLETE: ", data["concept:name"].cat.categories.get_loc("W_Valideren aanvraag-COMPLETE") + 1)
    print("W_Nabellen offertes-START: ", data["concept:name"].cat.categories.get_loc("W_Nabellen offertes-START") + 1)
    print("O_SENT_BACK-COMPLETE: ", data["concept:name"].cat.categories.get_loc("O_SENT_BACK-COMPLETE") + 1)
    print("O_CANCELLED-COMPLETE: ", data["concept:name"].cat.categories.get_loc("O_CANCELLED-COMPLETE") + 1)
    print("A_ACCEPTED-COMPLETE: ", data["concept:name"].cat.categories.get_loc("A_ACCEPTED-COMPLETE") + 1)
    data["concept:name"] = data["concept:name"].cat.codes + 1
    vocab_sizes["concept:name"] = data["concept:name"].max()

    data["org:resource"] = pd.Categorical(data["org:resource"])
    data["org:resource"] = data["org:resource"].cat.codes + 1
    vocab_sizes["org:resource"] = data["org:resource"].max()

    # Numerical values
    data["case:AMOUNT_REQ"] = data["case:AMOUNT_REQ"].ffill()
    data["case:AMOUNT_REQ"] = scaler_ar.fit_transform(data[["case:AMOUNT_REQ"]])
    data["elapsed_time"] = data["elapsed_time"].fillna(0)
    data["elapsed_time"] = scaler_elapsed.fit_transform(data[["elapsed_time"]])
    data["time_since_previous"] = data["time_since_previous"].fillna(0)
    data["time_since_previous"] = scaler_time_prev.fit_transform(data[["time_since_previous"]])

    scalers = {
        "case:AMOUNT_REQ": scaler_ar,
        "elapsed_time": scaler_elapsed,
        "time_since_previous": scaler_time_prev
    }

    return create_ngrams(data, train_ids, val_ids, test_ids), vocab_sizes, scalers