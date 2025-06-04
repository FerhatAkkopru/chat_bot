from dotenv import load_dotenv
import os

print("Dosya yükleniyor...")
load_dotenv("secret.env")

api_key = os.getenv("OPENAI_API_KEY")
print(f"API Anahtarı: {api_key if api_key else 'Bulunamadı'}") 