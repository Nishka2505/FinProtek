# explain.py
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
#client = Groq(api_key=os.getenv("GROQ_API_KEY"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=8.0)
def explain_flag(txn_row):
    prompt = f"""
A financial transaction has been flagged as potentially fraudulent by a machine learning model.

Transaction details:
- Amount deviation from user's average: {txn_row['amount_deviation']:.2f}x
- New/unrecognized device used: {"Yes" if txn_row['new_device'] else "No"}
- New/unrecognized location: {"Yes" if txn_row['new_location'] else "No"}
- Transaction at unusual hour (midnight-5am): {"Yes" if txn_row['odd_hour'] else "No"}
- Transactions by this user in the last hour: {txn_row['velocity']}

Write a clear, structured explanation for a fraud analyst reviewing this alert. Respond in exactly this format:

RISK SUMMARY: (one sentence, plain English, why this looks suspicious)
KEY FACTORS: (2-3 bullet points, most important factor first, each factor stated with its specific value)
RECOMMENDED ACTION: (one short sentence - e.g. hold for manual review, verify via OTP, or auto-decline)

Only use the numbers given above. Do not invent transaction amounts, names, or details not listed.
Keep the whole response under 90 words.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    sample_txn = {
        "amount_deviation": 4.2,
        "new_device": 1,
        "new_location": 1,
        "odd_hour": 1,
        "velocity": 3
    }
    print(explain_flag(sample_txn))