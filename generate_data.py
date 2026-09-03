# generate_data.py
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()
random.seed(42)
np.random.seed(42)

NUM_USERS = 100
NUM_TRANSACTIONS = 1000
FRAUD_RATE = 0.08

users = []
for i in range(NUM_USERS):
    users.append({
        "user_id": f"u{i+1}",
        "avg_amount": round(np.random.uniform(500, 5000), 2),
        "home_location": fake.city(),
        "known_device": f"device_{i+1}_A"
    })

users_df = pd.DataFrame(users)

transactions = []
start_time = datetime(2026, 8, 1, 0, 0, 0)

for i in range(NUM_TRANSACTIONS):
    user = users_df.sample(1).iloc[0]
    is_fraud = random.random() < FRAUD_RATE
    txn_time = start_time + timedelta(minutes=random.randint(0, 30000))

    if is_fraud:
        # Fraud: elevated but NOT guaranteed signals - creates realistic ambiguity
        amount = round(user["avg_amount"] * random.uniform(1.5, 8), 2)
        device = f"device_unknown_{random.randint(1000,9999)}" if random.random() < 0.7 else user["known_device"]
        location = fake.city() if random.random() < 0.6 else user["home_location"]
        if random.random() < 0.5:
            hour = random.choice([0, 1, 2, 3, 4])
        else:
            hour = random.choice(range(6, 23))
        txn_time = txn_time.replace(hour=hour)
    else:
        # Normal: occasionally has one "risky-looking" trait anyway (real life is messy)
        amount = round(user["avg_amount"] * random.uniform(0.7, 1.3), 2)
        if random.random() < 0.08:
            amount = round(amount * random.uniform(1.5, 2.2), 2)  # occasional big legit purchase
        device = user["known_device"] if random.random() < 0.9 else f"device_new_{random.randint(1000,9999)}"
        location = user["home_location"] if random.random() < 0.9 else fake.city()
        hour = random.choice(range(7, 23)) if random.random() < 0.9 else random.choice([0, 1, 2, 3])
        txn_time = txn_time.replace(hour=hour)

    transactions.append({
        "txn_id": f"t{i+1}",
        "user_id": user["user_id"],
        "amount": amount,
        "timestamp": txn_time,
        "device_id": device,
        "location": location,
        "merchant_id": f"m{random.randint(1,20)}",
        "is_fraud": int(is_fraud)
    })

txn_df = pd.DataFrame(transactions)
txn_df = txn_df.sort_values("timestamp").reset_index(drop=True)

users_df.to_csv("data/user_profiles.csv", index=False)
txn_df.to_csv("data/transactions.csv", index=False)

print(f"Generated {len(users_df)} users and {len(txn_df)} transactions")
print(f"Fraud transactions: {txn_df['is_fraud'].sum()} ({txn_df['is_fraud'].mean()*100:.1f}%)")