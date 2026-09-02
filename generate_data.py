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
FRAUD_RATE = 0.08  # ~8% of transactions will be fraud

# Step 1: Create user profiles with a "normal" spending baseline
users = []
for i in range(NUM_USERS):
    users.append({
        "user_id": f"u{i+1}",
        "avg_amount": round(np.random.uniform(500, 5000), 2),
        "home_location": fake.city(),
        "known_device": f"device_{i+1}_A"
    })

users_df = pd.DataFrame(users)

# Step 2: Generate transactions
transactions = []
start_time = datetime(2026, 8, 1, 0, 0, 0)

for i in range(NUM_TRANSACTIONS):
    user = users_df.sample(1).iloc[0]
    is_fraud = random.random() < FRAUD_RATE

    txn_time = start_time + timedelta(minutes=random.randint(0, 30000))

    if is_fraud:
        # Fraud pattern: high amount deviation, new device, new location, odd hour
        amount = round(user["avg_amount"] * random.uniform(4, 10), 2)
        device = f"device_unknown_{random.randint(1000,9999)}"
        location = fake.city()
        hour = random.choice([1, 2, 3, 4])
        txn_time = txn_time.replace(hour=hour)
    else:
        # Normal pattern: close to average, known device/location, normal hours
        amount = round(user["avg_amount"] * random.uniform(0.7, 1.3), 2)
        device = user["known_device"]
        location = user["home_location"]
        hour = random.choice(range(7, 23))
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