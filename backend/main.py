# backend/main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from explain import explain_flag
from feature_engineering import compute_features
from sklearn.metrics import precision_score, recall_score, confusion_matrix

app = FastAPI(title="Fraud Spike Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/fraud_model.pkl")
DEFAULT_THRESHOLD = 0.6


def get_risk_tier(score):
    if score < 0.3:
        return "Low"
    elif score < 0.6:
        return "Medium"
    elif score < 0.85:
        return "High"
    else:
        return "Critical"


@app.get("/")
def home():
    return {"message": "Fraud Spike Detector API is running"}


@app.post("/predict")
def predict(txn: dict):
    X = pd.DataFrame([txn])[["amount_deviation", "new_device", "new_location", "odd_hour", "velocity"]]
    fraud_score = model.predict_proba(X)[0][1]
    tier = get_risk_tier(fraud_score)

    result = {
        "fraud_score": round(float(fraud_score), 3),
        "risk_tier": tier,
        "flagged": bool(fraud_score > DEFAULT_THRESHOLD)
    }

    if result["flagged"]:
        try:
            result["explanation"] = explain_flag(txn)
        except Exception as e:
            result["explanation"] = "AI explanation unavailable (connection issue)."
            print(f"Explanation failed: {e}")

    return result


@app.get("/analyze")
def analyze_all(threshold: float = DEFAULT_THRESHOLD):
    txn_df = pd.read_csv("data/transactions.csv")
    users_df = pd.read_csv("data/user_profiles.csv")

    feature_df = compute_features(txn_df, users_df)
    feature_df = feature_df.merge(txn_df[["txn_id", "amount"]], on="txn_id", how="left")

    X = feature_df[["amount_deviation", "new_device", "new_location", "odd_hour", "velocity"]]
    y_true = feature_df["is_fraud"]

    fraud_scores = model.predict_proba(X)[:, 1]
    feature_df["fraud_score"] = fraud_scores
    feature_df["flagged"] = fraud_scores > threshold
    feature_df["risk_tier"] = feature_df["fraud_score"].apply(get_risk_tier)

    precision = precision_score(y_true, feature_df["flagged"], zero_division=0)
    recall = recall_score(y_true, feature_df["flagged"], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, feature_df["flagged"]).ravel()

    money_protected = feature_df[(feature_df["flagged"]) & (feature_df["is_fraud"] == 1)]["amount"].sum()
    total = len(feature_df)
    safe_percentage = ((total - int(feature_df["flagged"].sum())) / total) * 100

    tier_counts = feature_df["risk_tier"].value_counts().to_dict()
    tier_distribution = {
        "Low": tier_counts.get("Low", 0),
        "Medium": tier_counts.get("Medium", 0),
        "High": tier_counts.get("High", 0),
        "Critical": tier_counts.get("Critical", 0),
    }

    flagged_txns = feature_df[feature_df["flagged"]].sort_values("fraud_score", ascending=False).head(5)
    flagged_list = []
    for _, row in flagged_txns.iterrows():
        try:
            explanation = explain_flag(row)
        except Exception as e:
            explanation = "AI explanation unavailable (connection issue). Score computed from behavioral features only."
            print(f"Explanation failed for {row['txn_id']}: {e}")

        flagged_list.append({
            "txn_id": row["txn_id"],
            "amount": round(float(row["amount"]), 2),
            "fraud_score": round(float(row["fraud_score"]), 3),
            "risk_tier": row["risk_tier"],
            "amount_deviation": round(float(row["amount_deviation"]), 2),
            "new_device": bool(row["new_device"]),
            "new_location": bool(row["new_location"]),
            "odd_hour": bool(row["odd_hour"]),
            "velocity": int(row["velocity"]),
            "explanation": explanation
        })

    return {
        "threshold": threshold,
        "total_transactions": total,
        "flagged_count": int(feature_df["flagged"].sum()),
        "actual_fraud_count": int(y_true.sum()),
        "safe_percentage": round(safe_percentage, 1),
        "money_protected": round(float(money_protected), 2),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "fp_cost": int(fp) * 50,
        "fn_cost": int(fn) * 3000,
        "tier_distribution": tier_distribution,
        "flagged_transactions": flagged_list
    }