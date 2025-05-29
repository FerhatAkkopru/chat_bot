import streamlit as st
import requests

API_URL = "http://localhost:8000/webhook"  # FastAPI çalışıyorsa bu adresi kullanıyoruz

st.title("Akıllı Teknik Soru-Cevap Sistemi")

question = st.text_area("Sorunuzu yazınız:")

if st.button("Soruyu Gönder"):
    if not question.strip():
        st.warning("Lütfen bir soru giriniz.")
    else:
        with st.spinner("Cevap aranıyor..."):
            try:
                response = requests.post(API_URL, json={"question": question})
                if response.status_code == 200:
                    answer = response.json().get("answer", "Cevap alınamadı.")
                    st.success(answer)
                else:
                    st.error(f"API hatası: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Sunucuya bağlanılamadı: {e}")
