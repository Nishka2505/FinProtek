# feature_engineering.py
import pandas as pd
import numpy as np

def compute_features(txn_df, users_df):
    txn_df = txn_df.copy()
    txn_df["timestamp"] = pd.to_datetime(txn_df["timestamp"])
    txn_df = txn_df.sort_values("timestamp").reset_index(drop=True)

    features = []

    for idx, row in txn_df.iterrows():
        user_id = row["user_id"]
        user_profile = users_df[users_df["user_id"] == user_id].iloc[0]

        # Feature 1: Amount deviation from user's average
        amount_deviation = (row["amount"] - user_profile["avg_amount"]) / user_profile["avg_amount"]

        # Feature 2: New device flag
        new_device = 1 if row["device_id"] != user_profile["known_device"] else 0

        # Feature 3: New location flag
        new_location = 1 if row["location"] != user_profile["home_location"] else 0

        # Feature 4: Odd hour flag (midnight - 5am considered unusual)
        hour = row["timestamp"].hour
        odd_hour = 1 if hour < 5 else 0

        # Feature 5: Velocity — transactions by same user in the last 1 hour
        past_hour = row["timestamp"] - pd.Timedelta(hours=1)
        recent_txns = txn_df[
            (txn_df["user_id"] == user_id) &
            (txn_df["timestamp"] > past_hour) &
            (txn_df["timestamp"] <= row["timestamp"])
        ]
        velocity = len(recent_txns)

        features.append({
            "txn_id": row["txn_id"],
            "amount_deviation": amount_deviation,
            "new_device": new_device,
            "new_location": new_location,
            "odd_hour": odd_hour,
            "velocity": velocity,
            "is_fraud": row["is_fraud"]
        })

    return pd.DataFrame(features)

if __name__ == "__main__":
    txn_df = pd.read_csv("data/transactions.csv")
    users_df = pd.read_csv("data/user_profiles.csv")

    feature_df = compute_features(txn_df, users_df)
    feature_df.to_csv("data/features.csv", index=False)

    print(f"Computed features for {len(feature_df)} transactions")
    print(feature_df.head())