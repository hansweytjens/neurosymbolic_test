import pandas as pd
import os

print("Reading raw data...")
df = pd.read_csv('/workspace/data/prepare/bpi12_raw.csv', dtype={'org:resource': str})
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format='ISO8601', utc=True)
df = df.sort_values(['case:concept:name', 'time:timestamp']).reset_index(drop=True)

# Combine concept:name + lifecycle:transition -> e.g. "A_SUBMITTED-COMPLETE"
df['concept:name'] = df['concept:name'] + '-' + df['lifecycle:transition']
df = df.drop(columns=['lifecycle:transition'])

# Clean org:resource: "112.0" -> "112"
df['org:resource'] = df['org:resource'].apply(
    lambda x: str(int(float(x))) if pd.notna(x) and x not in ('nan', 'None') else 'UNKNOWN'
)

# Label: 1 if A_ACCEPTED-COMPLETE appears anywhere in the case
accepted_cases = set(df[df['concept:name'] == 'A_ACCEPTED-COMPLETE']['case:concept:name'])
df['label'] = df['case:concept:name'].isin(accepted_cases).astype(int)

# Temporal features
df['elapsed_time'] = df.groupby('case:concept:name')['time:timestamp'].transform(
    lambda x: (x - x.min()).dt.total_seconds() / 60
)
df['time_since_previous'] = (
    df.groupby('case:concept:name')['time:timestamp']
    .diff().dt.total_seconds().fillna(0) / 60
)

os.makedirs('/workspace/data_processed', exist_ok=True)
df.to_csv('/workspace/data_processed/bpi12.csv', index=False)

print("Done! Saved to data_processed/bpi12.csv")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nLabel distribution:\n{df.groupby('case:concept:name')['label'].first().value_counts()}")
print(f"\nSample concept:name values:\n{df['concept:name'].unique()[:10]}")
