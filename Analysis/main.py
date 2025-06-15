from pathlib import Path
import pandas as pd

# Get the root directory (2 levels up from this file)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Define the data directory
DATA_DIR = ROOT_DIR / "data"

# Load datasets
claims = pd.read_csv(DATA_DIR / "fact_claims.csv")
members = pd.read_csv(DATA_DIR / "member_dimension.csv")
diagnoses = pd.read_csv(DATA_DIR / "diagnosis_dimension.csv")
providers = pd.read_csv(DATA_DIR / "provider_dimension.csv")

# Merge all into one DataFrame
df = claims.merge(members, on="member_id", how="left") \
           .merge(diagnoses, on="diagnosis_code", how="left") \
           .merge(providers, on="provider_id", how="left")

# Create age_group column
bins = [0, 25, 35, 45, 60, 120]
labels = ["18–25", "26–35", "36–45", "46–60", "60+"]
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Outlier detection (Z-score)
df['z_score'] = (df['claim_amount'] - df['claim_amount'].mean()) / df['claim_amount'].std()
outliers = df[df['z_score'].abs() > 3]

# Top diagnosis per age group

age_diag = df.groupby(['age_group', 'diagnosis_description'], observed=True)['claim_id'].count().reset_index()
top_diag = (
    age_diag.sort_values(['age_group', 'claim_id'], ascending=[True, False])
    .groupby('age_group', observed=True)
    .head(1)
)


print("Top diagnoses per age group:")
print(top_diag)
