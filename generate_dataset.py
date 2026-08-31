import pandas as pd
import numpy as np

np.random.seed(42)

n = 10000

transaction_types = [
    "CASH_OUT",
    "TRANSFER",
    "PAYMENT",
    "CASH_IN"
]

data = {
    "transaction_amount": np.random.exponential(50000, n),
    "transaction_type": np.random.choice(
        transaction_types, n, p=[0.30, 0.30, 0.25, 0.15]
    ),
    "old_balance": np.random.uniform(0, 500000, n),
    "transaction_frequency": np.random.poisson(3, n),
    "account_age_days": np.random.randint(30, 2000, n),
    "hour": np.random.randint(0, 24, n),
    "device_change": np.random.choice([0, 1], n, p=[0.9, 0.1]),
    "location_change": np.random.choice([0, 1], n, p=[0.92, 0.08]),
}

df = pd.DataFrame(data)

# Create fraud probability
fraud_score = (
    (df["transaction_amount"] > 150000) * 0.30
    + (df["transaction_frequency"] > 8) * 0.20
    + (df["device_change"] == 1) * 0.20
    + (df["location_change"] == 1) * 0.15
    + ((df["hour"] < 5) | (df["hour"] > 23)) * 0.10
    + (df["account_age_days"] < 90) * 0.05
)

# Add randomness
fraud_probability = fraud_score + np.random.normal(0, 0.08, n)

df["is_fraud"] = (fraud_probability > 0.35).astype(int)

# Save dataset
df.to_csv("data/transactions.csv", index=False)

print("Dataset generated successfully!")
print(f"Total transactions: {len(df)}")
print(f"Fraudulent transactions: {df['is_fraud'].sum()}")
print(f"Legitimate transactions: {(df['is_fraud'] == 0).sum()}")