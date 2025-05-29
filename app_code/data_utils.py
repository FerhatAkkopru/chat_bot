import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import uuid
import os

# Dosya yolları (webhook.py'nin bulunduğu dizine göre ../data/ olmalı
# ya da proje kökünden çalıştırılıyorsa data/ olabilir.
# Mevcut yapıya göre data/ varsayıyoruz, app_code ve data kardeş dizinler)
BASE_DATA_PATH = "data"
DATA_JSON = os.path.join(BASE_DATA_PATH, "datas.json")
EMBEDDINGS_FILE = os.path.join(BASE_DATA_PATH, "embeddings.npy")
METADATA_FILE = os.path.join(BASE_DATA_PATH, "metadata.pkl")
FAISS_FILE = os.path.join(BASE_DATA_PATH, "faiss_index.index")

# Modelin tekrar yüklenmesini önlemek için global bir değişken
_model_instance = None

def get_sentence_transformer_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return _model_instance

def _ensure_data_dir_exists():
    os.makedirs(BASE_DATA_PATH, exist_ok=True)

def load_data():
    _ensure_data_dir_exists()
    if not os.path.exists(DATA_JSON):
        return []
    try:
        with open(DATA_JSON, "r", encoding="utf-8") as f:
            content = f.read()
            if not content: # Dosya boşsa
                return []
            return json.loads(content)
    except json.JSONDecodeError:
        print(f"⚠️ '{DATA_JSON}' dosyası çözümlenemedi, boş liste döndürülüyor.")
        return [] # Bozuk JSON durumunda boş liste döndür

def save_data(data):
    _ensure_data_dir_exists()
    with open(DATA_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_metadata():
    _ensure_data_dir_exists()
    if not os.path.exists(METADATA_FILE) or os.path.getsize(METADATA_FILE) == 0:
        return []
    try:
        with open(METADATA_FILE, "rb") as f:
            return pickle.load(f)
    except pickle.UnpicklingError:
        print(f"⚠️ '{METADATA_FILE}' dosyası çözümlenemedi, boş liste döndürülüyor.")
        return [] # Bozuk pickle durumunda
    except EOFError: # Dosya beklenenden önce biterse (boş veya çok kısa)
        print(f"⚠️ '{METADATA_FILE}' dosyası okunurken EOFError, boş liste döndürülüyor.")
        return []


def save_metadata(metadata):
    _ensure_data_dir_exists()
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

def load_embeddings():
    _ensure_data_dir_exists()
    model = get_sentence_transformer_model() # Model embedding boyutunu almak için
    embedding_dim = model.get_sentence_embedding_dimension()
    if not os.path.exists(EMBEDDINGS_FILE):
        # FAISS için float32 tipinde boş bir 2D array döndür
        return np.empty((0, embedding_dim), dtype=np.float32)
    try:
        embeddings = np.load(EMBEDDINGS_FILE)
        # Yüklenen embeddinglerin beklenen boyutta olup olmadığını kontrol et
        if embeddings.ndim == 2 and embeddings.shape[1] == embedding_dim:
            return embeddings.astype(np.float32)
        elif embeddings.size == 0: # Dosya var ama boş numpy array'i içeriyor
             return np.empty((0, embedding_dim), dtype=np.float32)
        else:
            print(f"⚠️ '{EMBEDDINGS_FILE}' içindeki embeddinglerin boyutu ({embeddings.shape}) beklenenden farklı ({embedding_dim}). Boş array döndürülüyor.")
            return np.empty((0, embedding_dim), dtype=np.float32)
    except ValueError as e: # Numpy dosyası bozuksa
        print(f"⚠️ '{EMBEDDINGS_FILE}' yüklenirken hata: {e}. Boş array döndürülüyor.")
        return np.empty((0, embedding_dim), dtype=np.float32)


def save_embeddings(embeddings):
    _ensure_data_dir_exists()
    np.save(EMBEDDINGS_FILE, embeddings.astype(np.float32))

def update_faiss_index(embeddings):
    _ensure_data_dir_exists()
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2 or embeddings.shape[0] == 0:
        print("⚠️ FAISS index oluşturmak için geçerli embedding bulunamadı.")
        if os.path.exists(FAISS_FILE): # Geçersiz durumda eski indeksi sil
            try:
                os.remove(FAISS_FILE)
                print(f"🗑️ Eski FAISS index dosyası '{FAISS_FILE}' silindi.")
            except OSError as e:
                print(f"🚨 Eski FAISS index dosyası '{FAISS_FILE}' silinirken hata: {e}")
        return

    dimension = embeddings.shape[1]
    # FAISS için C-contiguous ve float32 olduğundan emin ol
    embeddings_for_faiss = np.ascontiguousarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings_for_faiss) # Veritabanı embeddinglerini normalize et
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_for_faiss)
    faiss.write_index(index, FAISS_FILE)
    print(f"✅ FAISS index güncellendi ve '{FAISS_FILE}' dosyasına kaydedildi. Toplam {index.ntotal} vektör.")


def get_answer_by_id(record_id):
    metadata = load_metadata()
    for item in metadata:
        if item.get("id") == record_id: # Daha güvenli erişim için .get() kullan
            return item.get("cevap") # Cevap yoksa None döner
    return None

def add_question(soru, cevap):
    model = get_sentence_transformer_model() # Paylaşılan model örneğini al
    new_id = str(uuid.uuid4())
    # Sorunun string olduğundan emin ol, encode için liste içinde ver
    new_embedding_single = model.encode([str(soru)], convert_to_numpy=True).astype(np.float32)

    data = load_data()
    metadata = load_metadata()
    current_embeddings = load_embeddings()

    data.append({"id": new_id, "soru": soru, "cevap": cevap})
    metadata.append({"id": new_id, "soru": soru, "cevap": cevap})

    if current_embeddings.shape[0] == 0:
        updated_embeddings = new_embedding_single # Zaten (1, dim) şeklinde olmalı
    else:
        # new_embedding_single'ın (1, dim) şeklinde olduğundan emin ol
        if new_embedding_single.ndim == 1: # encode bazen 1D dönebilir, 2D'ye çevir
             new_embedding_single = np.expand_dims(new_embedding_single, axis=0)
        updated_embeddings = np.vstack([current_embeddings, new_embedding_single])

    save_data(data)
    save_metadata(metadata)
    save_embeddings(updated_embeddings)
    update_faiss_index(updated_embeddings) # Bu fonksiyon artık boş embeddingleri de ele alıyor

    print(f"✅ Yeni soru eklendi: '{soru}' (id: {new_id})")