import openai
from dotenv import load_dotenv
import os
import sys
from pathlib import Path


# Ana dizindeki secret.env dosyasının tam yolunu al
base_dir = Path(__file__).resolve().parent.parent
env_path = base_dir / "secret.env"

print(f"Çevre değişkenleri dosyası yükleniyor: {env_path}")
if not env_path.exists():
    raise FileNotFoundError(f"secret.env dosyası bulunamadı: {env_path}")

load_dotenv(env_path)
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(f"OPENAI_API_KEY bulunamadı. Lütfen {env_path} dosyasını kontrol edin.")

print("API anahtarı başarıyla yüklendi.")
client = openai.OpenAI(api_key=api_key)

def get_gpt_answer(question):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Makine öğrenmesi alanında uzman bir asistansın. Açık ve sade cevaplar ver."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content.strip()