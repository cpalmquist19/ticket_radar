import pandas as pd

df = pd.read_csv(r'C:\Code\ml\ticket_intent\training\aa_dataset-tickets-multi-lang-5-2-50-version_cleaned.csv')

print("=" * 80)
print("PROBLEM SAMPLES (first 3)")
print("=" * 80)
problem_samples = df[df['type'] == 'Problem'].head(3)
for idx, (_, row) in enumerate(problem_samples.iterrows(), 1):
    print(f"\n--- Problem Sample {idx} ---")
    print(f"Subject: {row['subject']}")
    print(f"Body (first 300 chars): {row['body'][:300]}...")
    print(f"Queue: {row.get('queue', 'N/A')}")

print("\n\n" + "=" * 80)
print("INCIDENT SAMPLES (first 3)")
print("=" * 80)
incident_samples = df[df['type'] == 'Incident'].head(3)
for idx, (_, row) in enumerate(incident_samples.iterrows(), 1):
    print(f"\n--- Incident Sample {idx} ---")
    print(f"Subject: {row['subject']}")
    print(f"Body (first 300 chars): {row['body'][:300]}...")
    print(f"Queue: {row.get('queue', 'N/A')}")

print("\n\n" + "=" * 80)
print("LABEL DISTRIBUTION BY QUEUE")
print("=" * 80)
print(df.groupby(['queue', 'type']).size().unstack(fill_value=0))

