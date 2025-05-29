# Gelişmiş Soru-Cevap Chat Botu

Bu proje, kullanıcılardan gelen teknik soruları yanıtlayan ve zamanla yeni bilgiler öğrenerek kendini geliştiren bir soru-cevap chatbot'udur. Sistem, anlamsal benzerlik araması için FAISS ve Sentence Transformers kullanır ve yeni sorular için OpenAI GPT modelinden yararlanır.

## Proje Yapısı

```
/chat_bot
|-- /app_code                 # Ana uygulama mantığının bulunduğu dizin
|   |-- __init__.py
|   |-- anahtar_kelimeler.py  # Teknik soruları belirlemek için anahtar kelimeler
|   |-- data_utils.py         # Veri yükleme, kaydetme ve FAISS işlemleri için yardımcı fonksiyonlar
|   |-- is_tech.py            # Bir sorunun teknik olup olmadığını belirleyen fonksiyon
|   |-- openai_integration.py # OpenAI API ile etkileşim
|   `-- webhook.py            # FastAPI tabanlı ana uygulama ve webhook endpoint'i
|
|-- /data                     # Veri dosyalarının ve ön işlenmiş çıktıların bulunduğu dizin
|   |-- __init__.py
|   |-- datas.json            # Soru-cevap çiftlerini içeren ana veri dosyası
|   |-- embeddings.npy        # Soru embedding'lerinin saklandığı dosya (oluşturulur)
|   |-- faiss_index.index     # FAISS arama indeksinin saklandığı dosya (oluşturulur)
|   |-- metadata.pkl          # Soru ID'leri, soruları ve cevapları içeren metadata (oluşturulur)
|   `-- process_data.py       # datas.json'dan embedding ve FAISS indeksi oluşturan betik
|
|-- .gitignore                # Git tarafından takip edilmeyecek dosyalar
|-- README.md                 # Bu dosya
`-- secret.env                # API anahtarları gibi gizli bilgileri içeren dosya 
```

## Kurulum

1.  **Depoyu Klonlayın:**
    ```bash
    git clone <depo_url>
    cd chat_bot
    ```

2.  **Sanal Ortam Oluşturun ve Aktifleştirin (Önerilir):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # macOS/Linux için
    # venv\Scripts\activate   # Windows için
    ```

3.  **Gerekli Kütüphaneleri Yükleyin:**
    Proje için gerekli kütüphaneleri içeren bir `requirements.txt` dosyası oluşturmanız ve aşağıdaki gibi yüklemeniz önerilir. Eğer `requirements.txt` dosyanız yoksa, temel kütüphaneler şunlardır:
    ```bash
    pip install fastapi uvicorn sentence-transformers faiss-cpu numpy openai python-dotenv
    ```
    *(Not: `faiss-cpu` yerine GPU desteği için `faiss-gpu` kullanabilirsiniz.)*

    **Örnek `requirements.txt` içeriği:**
    ```txt
    fastapi
    uvicorn[standard]
    sentence-transformers
    faiss-cpu
    numpy
    openai
    python-dotenv
    # nltk # Eğer is_tech.py'de lemmatization kullanılıyorsa
    ```

4.  **Ortam Değişkenlerini Ayarlayın:**
    Proje kök dizininde `secret.env` adında bir dosya oluşturun ve OpenAI API anahtarınızı bu dosyaya ekleyin:
    ```env
    OPENAI_API_KEY="sk-sizin_openai_api_anahtarınız"
    ```
    `openai_integration.py` dosyasındaki `load_dotenv()` satırının yorumunu kaldırın veya API anahtarını doğrudan kod içinde (güvenlik açısından önerilmez) ya da ortam değişkeni olarak ayarlayın. Mevcut kodunuzda API anahtarı doğrudan `openai_integration.py` içine yazılmış görünüyor. Güvenlik için bunu `secret.env` dosyasına taşımak ve `python-dotenv` ile yüklemek daha iyidir.

    `openai_integration.py` dosyasını şu şekilde güncelleyebilirsiniz:
    ````python
    # filepath: app_code/openai_integration.py
    # ...
    from dotenv import load_dotenv
    import os

    # Proje kök dizinindeki secret.env dosyasını yükle
    # Bu dosyanın webhook.py'nin bir üst dizininde olduğunu varsayıyoruz.
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'secret.env')
    load_dotenv(dotenv_path=dotenv_path)

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # client = openai.OpenAI(api_key="sk-proj-28hicL...") # Bu satırı kaldırın veya yorumlayın
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY ortam değişkeni bulunamadı. Lütfen secret.env dosyasını kontrol edin.")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    # ...
    ````

## Kullanım

### 1. Veri Ön İşleme (Embedding ve FAISS İndeksi Oluşturma)

Eğer `data/datas.json` dosyanızda başlangıç soru-cevap verileriniz varsa, bu verileri işleyerek embedding'leri ve FAISS arama indeksini oluşturmanız gerekir. Bu işlem, sistemin ilk defa çalıştırılmadan önce veya `datas.json` güncellendiğinde yapılmalıdır.

Proje kök dizinindeyken aşağıdaki komutu çalıştırın:
```bash
python data/process_data.py
```
Bu komut, `data/datas.json` dosyasını okuyacak ve `data/` dizini içine `embeddings.npy`, `metadata.pkl` ve `faiss_index.index` dosyalarını oluşturacaktır. Eğer `datas.json` boşsa veya mevcut değilse, bu adım atlanabilir ve sistem ilk sorudan itibaren öğrenmeye başlayabilir.

### 2. Ana Uygulamayı Çalıştırma (Webhook)

FastAPI tabanlı webhook uygulamasını başlatmak için proje kök dizinindeyken aşağıdaki komutu çalıştırın:
```bash
uvicorn app_code.webhook:app --reload --host 0.0.0.0 --port 8000
```
`--reload` seçeneği, kodda değişiklik yaptığınızda sunucunun otomatik olarak yeniden başlatılmasını sağlar (geliştirme aşamasında kullanışlıdır).

Sunucu başlatıldıktan sonra, genellikle `http://localhost:8000/docs` adresinden API dokümantasyonuna (Swagger UI) erişebilirsiniz.

### 3. Soru Sorma

Uygulama çalışırken, `/webhook` endpoint'ine POST isteği göndererek soru sorabilirsiniz. İstek gövdesi aşağıdaki gibi bir JSON formatında olmalıdır:

```json
{
  "question": "Python'da bir liste nasıl sıralanır?"
}
```

Sistem, soruya mevcut bilgilerinden bir cevap bulmaya çalışacak, bulamazsa ve soruyu "teknik" olarak değerlendirirse OpenAI'ye sorup cevabı alacak ve bu yeni bilgiyi gelecekteki kullanımlar için kaydedecektir.

## Bileşenler

*   **`app_code/webhook.py`**: Kullanıcı isteklerini karşılayan ve ana iş mantığını yürüten FastAPI uygulaması.
*   **`app_code/data_utils.py`**: Veri dosyalarını (JSON, embeddings, metadata, FAISS index) okuma, yazma ve güncelleme işlemlerini yönetir.
*   **`app_code/openai_integration.py`**: OpenAI GPT modeli ile iletişim kurar.
*   **`app_code/is_tech.py`**: Bir sorunun teknik olup olmadığını anahtar kelimelerle belirler.
*   **`app_code/anahtar_kelimeler.py`**: `is_tech.py` tarafından kullanılan teknik anahtar kelimelerin listesini içerir.
*   **`data/process_data.py`**: Başlangıç verilerinden (`datas.json`) embedding ve FAISS indeksi oluşturur.
*   **`data/datas.json`**: Soru-cevap çiftlerinin saklandığı ana veri dosyası. Sistem yeni bilgiler öğrendikçe bu dosya güncellenir.
*   **`secret.env`**: API anahtarı gibi hassas bilgileri saklar. `.gitignore` ile Git'e eklenmesi engellenmiştir.

## Geliştirme Notları

*   Sistem, yeni sorular ve cevaplar öğrendikçe `data/datas.json`, `data/embeddings.npy`, `data/metadata.pkl` ve `data/faiss_index.index` dosyalarını dinamik olarak günceller.
*   `SIMILARITY_THRESHOLD` (webhook.py içinde) ve `KEYWORDS` (anahtar_kelimeler.py içinde) parametreleri, sistemin davranışını ayarlamak için değiştirilebilir.
