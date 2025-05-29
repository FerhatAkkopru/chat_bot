import openai
import numpy as np
import pickle
import faiss
from dotenv import load_dotenv
import os

# load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "secrets.env"))

# client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_gpt_answer(question):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Makine öğrenmesi alanında uzman bir asistansın. Açık ve sade cevaplar ver."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content.strip()