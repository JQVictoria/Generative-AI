import os, re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Gunakan model yang patuh pada System Prompt
MODEL_NAME = "google/gemini-2.5-flash"

SYSTEM = """Solve problems using this exact format:
<thinking>
Step-by-step reasoning here.
</thinking>
<answer>
The final answer only, no reasoning.
</answer>"""

try:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": "A RAG pipeline retrieves 5 documents, each 400 tokens. The query is 50 tokens. The model has a 4096 token limit for context. How many tokens remain for the response?"
            }
        ],
    )

    text = resp.choices[0].message.content if resp and resp.choices else ""

    # Extract sections
    thinking = re.search(
        r"<thinking>(.*?)</thinking>",
        text,
        re.DOTALL | re.IGNORECASE
    )

    answer = re.search(
        r"<answer>(.*?)</answer>",
        text,
        re.DOTALL | re.IGNORECASE
    )

    print("Reasoning:", thinking.group(1).strip() if thinking else "not found")
    print("Answer:   ", answer.group(1).strip() if answer else "not found")

    # Debugging: Tampilkan output mentah jika regex masih gagal
    if not thinking or not answer:
        print("\n--- Raw Output dari Model ---")
        print(text)

except Exception as e:
    print(f"Error saat memanggil API: {e}")