import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("data/transactions.csv")

# Features
X = df.drop("is_fraud", axis=1)

# Target
y = df["is_fraud"]

# Categorical and numerical features
categorical_features = ["transaction_type"]

numerical_features = [
    "transaction_amount",
    "old_balance",
    "transaction_frequency",
    "account_age_days",
    "hour",
    "device_change",
    "location_change"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# Model
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train
print("Training model...")
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL RESULTS")
print("=" * 40)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(pipeline, "models/fraud_model.pkl")

print("\nModel saved to models/fraud_model.pkl")