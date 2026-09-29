import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Gunakan os.getenv agar lebih aman
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Gunakan model yang stabil di OpenRouter
MODEL_NAME = "google/gemini-2.5-flash"

# Without CoT - model jumps to answer, more likely to be wrong
DIRECT_PROMPT = "If a model costs $3.00 per million input tokens and $15.00 per million output tokens, and a request uses 2,400 input tokens and 800 output tokens, what is the total cost in USD?"

# With CoT - model reasons through each step
COT_PROMPT = """If a model costs $3.00 per million input tokens and $15.00 per million output tokens,
and a request uses 2,400 input tokens and 800 output tokens,
what is the total cost in USD?
Think through this step by step before giving the final answer."""

# Zero-shot CoT: just adding "think step by step"
ZERO_SHOT_COT = """Solve this problem. Think step by step, showing each calculation.
Finally, state: ANSWER: $X.XXXXXX
Problem: A pipeline makes 50 API calls per hour. Each call uses an average of 1,200 input tokens
and 400 output tokens. The model costs $3.00/M input and $15.00/M output.
What is the daily cost?"""

for label, prompt in [
    ("Direct", DIRECT_PROMPT),
    ("CoT", COT_PROMPT),
    ("Zero-shot CoT", ZERO_SHOT_COT)
]:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        if resp and resp.choices and len(resp.choices) > 0:
            print(f"=== {label} ===")
            content = resp.choices[0].message.content
            print(content[:300] if content else "Tidak ada output.")
            print()
        else:
            print(f"=== {label} ===")
            print(f"Gagal mendapatkan respon: {resp}\n")

    except Exception as e:
        print(f"=== {label} ===")
        print(f"Error saat memanggil API: {e}\n")