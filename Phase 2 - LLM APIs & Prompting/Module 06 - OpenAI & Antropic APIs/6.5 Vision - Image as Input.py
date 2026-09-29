import os, base64
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Menggunakan OpenRouter
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Gunakan model vision yang valid di OpenRouter
MODEL_VISION = "google/gemini-2.5-flash"


# Option A: URL (fastest)
def describe_image_url(url: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_VISION, 
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe what you see in this image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": url
                        }
                    }
                ]
            }]
        )

        if response and response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return f"Gagal mendapatkan respon. Respon dari API: {response}"

    except Exception as e:
        return f"Error saat memanggil API: {e}"


# Option B: base64 (for local files)
def describe_image_file(path: str) -> str:
    try:
        data = Path(path).read_bytes()
        b64 = base64.standard_b64encode(data).decode()
        ext = Path(path).suffix.lstrip(".").lower()
        media_type = f"image/{ext}"  # image/png, image/jpeg, image/webp, image/gif

        response = client.chat.completions.create(
            model=MODEL_VISION,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64}"
                        }
                    }
                ]
            }]
        )

        if response and response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return f"Gagal mendapatkan respon. Respon dari API: {response}"

    except Exception as e:
        return f"Error saat memanggil API: {e}"


# Usage:
print("Sedang menganalisis gambar dari Unsplash...")
text = describe_image_url("https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=800")
print("\nHasil Analisis:")
print(text)