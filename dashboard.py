# dashboard.py
import streamlit as st
import pandas as pd
import joblib
from feature_engineering import compute_features
from explain import explain_flag
from sklearn.metrics import precision_score, recall_score, confusion_matrix

st.set_page_config(page_title="Fraud Spike Detector", layout="wide")
st.title("🛡️ AI Risk Manager — Fraud Spike Detector")

model = joblib.load("models/fraud_model.pkl")
THRESHOLD = 0.6

# Load data
txn_df = pd.read_csv("data/transactions.csv")
users_df = pd.read_csv("data/user_profiles.csv")

st.subheader("Step 1: Feature Engineering")
if st.button("Run Feature Extraction"):
    with st.spinner("Computing features for all transactions..."):
        feature_df = compute_features(txn_df, users_df)
        st.session_state["feature_df"] = feature_df
    st.success(f"Computed features for {len(feature_df)} transactions")

if "feature_df" in st.session_state:
    feature_df = st.session_state["feature_df"]

    X = feature_df[["amount_deviation", "new_device", "new_location", "odd_hour", "velocity"]]
    y_true = feature_df["is_fraud"]

    fraud_scores = model.predict_proba(X)[:, 1]
    feature_df["fraud_score"] = fraud_scores
    feature_df["flagged"] = fraud_scores > THRESHOLD

    st.subheader("Step 2: Model Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(feature_df))
    col2.metric("Flagged as Fraud", int(feature_df["flagged"].sum()))
    col3.metric("Actual Fraud (ground truth)", int(y_true.sum()))

    precision = precision_score(y_true, feature_df["flagged"])
    recall = recall_score(y_true, feature_df["flagged"])
    tn, fp, fn, tp = confusion_matrix(y_true, feature_df["flagged"]).ravel()

    st.subheader("Step 3: Honest Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{precision:.2%}")
    col2.metric("Recall", f"{recall:.2%}")
    col3.metric("False Positives", fp)
    col4.metric("False Negatives (missed fraud)", fn)

    fp_cost = fp * 50
    fn_cost = fn * 3000
    st.warning(f"💰 Estimated False Positive Cost: ₹{fp_cost}  |  Estimated False Negative Cost: ₹{fn_cost}")

    st.subheader("Step 4: Flagged Transactions with AI Explanation")
    flagged = feature_df[feature_df["flagged"]].head(5)

    for idx, row in flagged.iterrows():
        with st.expander(f"Transaction {row['txn_id']} — Fraud Score: {row['fraud_score']:.2f}"):
            explanation = explain_flag(row)
            st.write(explanation)
            st.json({
                "amount_deviation": round(row["amount_deviation"], 2),
                "new_device": bool(row["new_device"]),
                "new_location": bool(row["new_location"]),
                "odd_hour": bool(row["odd_hour"]),
                "velocity": int(row["velocity"])
            })