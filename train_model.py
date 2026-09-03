# train_model.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib

df = pd.read_csv("data/features.csv")

X = df[["amount_deviation", "new_device", "new_location", "odd_hour", "velocity"]]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Simpler, less confident base model — shallower trees, fewer of them,
# plus subsampling so it doesn't perfectly memorize the training patterns
base_model = xgb.XGBClassifier(
    n_estimators=40,
    max_depth=3,
    learning_rate=0.15,
    subsample=0.7,
    colsample_bytree=0.7,
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
    eval_metric="logloss"
)

# Calibrate probabilities so scores spread smoothly across 0-100%
# instead of collapsing to near-0 or near-1
model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("===== MODEL EVALUATION (held-out test set) =====")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}, True Negatives: {tn}")
print(f"Score range in test set: min={y_proba.min():.3f}, max={y_proba.max():.3f}, mean={y_proba.mean():.3f}")

COST_PER_FALSE_ALARM = 50
AVG_FRAUD_LOSS = 3000
fp_cost = fp * COST_PER_FALSE_ALARM
fn_cost = fn * AVG_FRAUD_LOSS

print(f"\nEstimated False Positive Cost: Rs.{fp_cost}")
print(f"Estimated False Negative Cost: Rs.{fn_cost}")
print(f"Total Estimated Cost: Rs.{fp_cost + fn_cost}")

joblib.dump(model, "models/fraud_model.pkl")
print("\nModel saved to models/fraud_model.pkl")