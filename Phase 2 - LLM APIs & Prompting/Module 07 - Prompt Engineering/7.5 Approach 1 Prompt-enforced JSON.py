import os, json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Gunakan model yang stabil dan patuh terhadap instruksi JSON
MODEL_NAME = "google/gemini-2.5-flash"

SYSTEM = """You are a data extractor. Extract information and return ONLY a JSON object.
No markdown, no explanation, no code fences. Raw JSON only.
Schema:
{
  "company": string,
  "founded": integer or null,
  "products": [string],
  "headquarters": string or null,
  "is_public": boolean
}"""

texts = [
    "Anthropic was founded in 2021 by Dario Amodei and others. It makes Claude AI models and is headquartered in San Francisco. It is a private company.",
    "OpenAI, founded in 2015, created ChatGPT and GPT-4. Based in San Francisco, it remains private despite a major Microsoft investment.",
]

def extract_company_info(text: str) -> dict:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=256,
            response_format={"type": "json_object"},  # Memaksa output berupa JSON
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": text}
            ],
        )

        if not hasattr(resp, 'choices') or not resp.choices:
            return {"error": "OpenRouter API is down or rate-limited."}

        content = resp.choices[0].message.content
        if content is None:
            return {"error": "API returned an empty response (None)"}

        raw = content.strip()

        # Strip any accidental markdown fences
        raw = (
            raw.removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        
        return json.loads(raw)
        
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw": raw}
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}


for text in texts:
    info = extract_company_info(text)
    print(json.dumps(info, indent=2))
    print()