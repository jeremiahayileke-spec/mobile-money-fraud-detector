import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load data
df = pd.read_csv("data/transactions.csv")

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Load trained model
model = joblib.load("models/fraud_model.pkl")

# Same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MOBILE MONEY FRAUD DETECTOR")
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraud"]
)

disp.plot()
plt.title("Mobile Money Fraud Detection - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# Fraud distribution
plt.figure(figsize=(7, 5))

df["is_fraud"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Distribution of Legitimate and Fraudulent Transactions")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")
plt.xticks(
    [0, 1],
    ["Legitimate", "Fraud"],
    rotation=0
)

plt.tight_layout()
plt.savefig("fraud_distribution.png", dpi=300)
plt.show()

print("\nCharts saved successfully.")