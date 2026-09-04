# explain.py
import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=20.0)

# Models tried in order — first one that returns non-empty content wins
_MODELS = [
    "groq/compound-mini",
    "groq/compound",
    "qwen/qwen3.8-27b",
]

def explain_flag(txn_row):
    flags = []
    if txn_row.get('new_device'):
        flags.append("transaction came from an unrecognized device")
    if txn_row.get('new_location'):
        flags.append("originated from a location never seen for this user")
    if txn_row.get('odd_hour'):
        flags.append("occurred between midnight and 5 am")
    dev = float(txn_row.get('amount_deviation', 1))
    if dev > 1.2:
        flags.append(f"amount is {dev:.1f}x above the user's normal average")
    vel = int(txn_row.get('velocity', 1))
    if vel >= 3:
        flags.append(f"{vel} transactions were made by this user in the past hour")

    flag_text = "; ".join(flags) if flags else "multiple behavioral anomalies detected"

    prompt = (
        "You are a fraud analyst AI. A machine learning model flagged a bank transaction.\n\n"
        "Transaction signals:\n"
        f"- Amount deviation: {dev:.2f}x the user's historical average\n"
        f"- New device: {'Yes' if txn_row.get('new_device') else 'No'}\n"
        f"- New location: {'Yes' if txn_row.get('new_location') else 'No'}\n"
        f"- Odd hour (midnight-5 am): {'Yes' if txn_row.get('odd_hour') else 'No'}\n"
        f"- Transactions in last hour: {vel}\n\n"
        "Respond in EXACTLY this format (no extra text, no markdown):\n\n"
        "RISK SUMMARY: <one sentence explaining why this looks fraudulent>\n"
        "KEY FACTORS:\n"
        "• <most important signal with its value>\n"
        "• <second signal with its value>\n"
        "• <third signal if applicable>\n"
        "RECOMMENDED ACTION: <one sentence — hold for review / verify via OTP / auto-decline>\n\n"
        "Rules: use only the numbers above, keep total response under 80 words, plain English."
    )

    last_error = None
    for model in _MODELS:
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=200,
                )
                content = response.choices[0].message.content.strip()
                if content:
                    return content
                last_error = Exception(f"Empty response from {model}")
            except Exception as e:
                last_error = e
                print(f"explain_flag [{model}] attempt {attempt+1} failed: {e}")
                time.sleep(0.8)

    raise last_error if last_error else Exception("All models failed")

if __name__ == "__main__":
    sample_txn = {
        "amount_deviation": 4.2,
        "new_device": 1,
        "new_location": 1,
        "odd_hour": 1,
        "velocity": 3
    }
    print(explain_flag(sample_txn))