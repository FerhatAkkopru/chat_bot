import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_JSON = "datas.json"
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.pkl"
FAISS_FILE = "faiss_index.index"

def load_data_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_faiss_index(embeddings):
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_FILE)
    print(f"FAISS index '{FAISS_FILE}' dosyasına kaydedildi.")

def main():
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    data = load_data_from_json(DATA_JSON)
    sorular = [item["soru"] for item in data]
    embeddings = model.encode(sorular, convert_to_numpy=True)
    metadata = [{"id": item["id"], "soru": item["soru"], "cevap": item["cevap"]} for item in data]

    np.save(EMBEDDINGS_FILE, embeddings)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    print(f"{len(sorular)} soru embedlendi ve metadata '{METADATA_FILE}' dosyasına kaydedildi.")
    build_faiss_index(embeddings)

if __name__ == "__main__":
    main()