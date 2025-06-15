from pathlib import Path
import pandas as pd

# Get the root directory (2 levels up from this file)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Define the data directory
DATA_DIR = ROOT_DIR / "data"

# Load datasets safely
claims = pd.read_csv(DATA_DIR / "fact_claims.csv")
members = pd.read_csv(DATA_DIR / "member_dimension.csv")
diagnoses = pd.read_csv(DATA_DIR / "diagnosis_dimension.csv")
providers = pd.read_csv(DATA_DIR / "provider_dimension.csv")

# Merge base model
df = claims.merge(members, on="member_id", how="left") \
           .merge(diagnoses, on="diagnosis_code", how="left") \
           .merge(providers, on="provider_id", how="left")

# Example 1: Outlier detection (Z-score)
df['z_score'] = (df['claim_amount'] - df['claim_amount'].mean()) / df['claim_amount'].std()
outliers = df[df['z_score'].abs() > 3]

# Example 2: Most common diagnosis per age group
age_diag = df.groupby(['Age Group', 'diagnosis_description'])['claim_id'].count().reset_index()
top_diag = age_diag.sort_values(['Age Group', 'claim_id'], ascending=[True, False]).groupby('Age Group').head(1)

print("Top diagnoses per age group:")
print(top_diag)
