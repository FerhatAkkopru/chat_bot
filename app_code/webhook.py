from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import time
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from app_code.is_tech import is_technical_question
from app_code.openai_integration import get_gpt_answer
from app_code.data_utils import add_question
import os # os modülünü import edin

app = FastAPI()

METADATA_FILE = "data/metadata.pkl"
FAISS_FILE = "data/faiss_index.index"
SIMILARITY_THRESHOLD = 0.8
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def process_user_question(user_question: str) -> str:
    start = time.time()
    # Kullanıcı sorusunun embedding'ini al ve normalize et
    query_embedding_orig = model.encode([user_question])[0].astype("float32")
    query_embedding_normalized = np.array([query_embedding_orig])
    faiss.normalize_L2(query_embedding_normalized) # Yerinde normalizasyon

    similarity = 0.0  # Varsayılan benzerlik
    matched_answer = None
    metadata = []

    try:
        if os.path.exists(FAISS_FILE) and os.path.exists(METADATA_FILE):
            index = faiss.read_index(FAISS_FILE)
            with open(METADATA_FILE, "rb") as f:
                # Pickle dosyasının boş olup olmadığını kontrol et
                if os.path.getsize(METADATA_FILE) > 0:
                    metadata = pickle.load(f)
                else:
                    print(f"⚠️ Metadata dosyası '{METADATA_FILE}' boş.")
                    metadata = []

            if index.ntotal > 0 and metadata: # İndeks ve metadata doluysa ara
                D, I = index.search(query_embedding_normalized, k=1)
                # I[0][0] değerinin metadata için geçerli bir indeks olduğundan emin ol
                if D.size > 0 and I.size > 0 and 0 <= I[0][0] < len(metadata):
                    similarity = D[0][0]
                    matched_item = metadata[I[0][0]]
                    matched_answer = matched_item.get("cevap")
                else:
                    print(f"⚠️ FAISS arama sonucu geçersiz veya metadata ile uyumsuz. Index: {I}, Metadata length: {len(metadata)}")
                    similarity = 0.0 # Eşleşme yok gibi davran
            else:
                print(f"⚠️ FAISS index '{FAISS_FILE}' boş veya metadata yüklenemedi/boş.")
                similarity = 0.0 # Eşleşme yok gibi davran
        else:
            print(f"⚠️ '{FAISS_FILE}' veya '{METADATA_FILE}' bulunamadı. Yeni soru olarak işlenecek.")
            # Dosyalar yoksa benzerlik düşük kabul edilir, matched_answer None kalır

    except FileNotFoundError:
        print(f"⚠️ '{FAISS_FILE}' veya '{METADATA_FILE}' yüklenirken FileNotFoundError. Yeni soru olarak işlenecek.")
        similarity = 0.0
    except Exception as e:
        print(f"🚨 FAISS/Metadata yüklenirken hata: {str(e)}")
        similarity = 0.0 # Genel bir hata durumunda da eşleşme yok gibi davran

    if similarity > SIMILARITY_THRESHOLD and matched_answer is not None:
        end = time.time()
        return f"✅ Benzer soru bulundu.\nCevap: {matched_answer}\n(Süre: {end - start:.2f} saniye)"
    else:
        if is_technical_question(user_question):
            gpt_answer = get_gpt_answer(user_question)
            # add_question fonksiyonu artık dosya yokluğunu ve boş dosyaları kendisi ele alacak
            add_question(user_question, gpt_answer)
            end = time.time()
            return f"⚠️ Benzer soru bulunamadı veya ilk çalıştırma. GPT'den cevap alındı ve kaydedildi:\n{gpt_answer}\n(Süre: {end - start:.2f} saniye)"
        else:
            return "Bu sistem yalnızca teknik sorulara cevap verir."

@app.post("/webhook")
async def webhook(request: Request):
    try:
        body = await request.json()
        question = body.get("question")
        if not question:
            return JSONResponse(content={"answer": "⚠️ Soru belirtilmedi."}, status_code=400)
        if not is_technical_question(question):
            return JSONResponse(content={"answer": "Bu sistem yalnızca teknik sorulara cevap verir."})
        answer = process_user_question(question)
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        print(f"🚨 Hata (webhook): {str(e)}")
        return JSONResponse(content={"answer": f"Hata oluştu: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    # data dizininin var olduğundan emin ol (data_utils.py de bunu yapacak ama burada da olabilir)
    os.makedirs("data", exist_ok=True)
    uvicorn.run("webhook:app", host="0.0.0.0", port=8000, reload=True)
