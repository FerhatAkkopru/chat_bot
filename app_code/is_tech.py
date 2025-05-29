from .anahtar_kelimeler import A as KEYWORDS # anahtar_kelimeler.py dosyasından A listesini KEYWORDS olarak import et

def is_technical_question(question: str) -> bool:
    question_lower = question.lower() # Soruyu bir kere küçük harfe çevir
    # KEYWORDS listesindeki her bir anahtar kelimenin (tekil veya n-gram)
    # küçük harfe çevrildiğinden emin olalım.
    return any(keyword.lower() in question_lower for keyword in KEYWORDS)