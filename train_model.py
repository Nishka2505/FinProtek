# train_model.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Load features
df = pd.read_csv("data/features.csv")

X = df[["amount_deviation", "new_device", "new_location", "odd_hour", "velocity"]]
y = df["is_fraud"]

# Split into train and held-out test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost classifier
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),  # handles class imbalance
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# Predict on held-out test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluate honestly
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("===== MODEL EVALUATION (held-out test set) =====")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}, True Negatives: {tn}")

# Cost estimation (this is "The Bar" requirement)
COST_PER_FALSE_ALARM = 50       # e.g., support ticket cost, customer friction
AVG_FRAUD_LOSS = 3000           # e.g., average money lost per missed fraud case

fp_cost = fp * COST_PER_FALSE_ALARM
fn_cost = fn * AVG_FRAUD_LOSS

print(f"\nEstimated False Positive Cost: Rs.{fp_cost}")
print(f"Estimated False Negative Cost: Rs.{fn_cost}")
print(f"Total Estimated Cost: Rs.{fp_cost + fn_cost}")

# Save model for later use
joblib.dump(model, "models/fraud_model.pkl")
print("\nModel saved to models/fraud_model.pkl")