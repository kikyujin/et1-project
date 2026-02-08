# translate.py
from google import genai
import os
from dotenv import load_dotenv

#load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- 設定 ---
model = 'gemini-2.5-flash-lite'  # ここでモデルを切り替える
text = 'おはようございます。今日はいい天気ですね。'

# --- 翻訳 ---
response = client.models.generate_content(
    model=model,
    contents=f'以下の文章を英語に翻訳してください。翻訳文のみを返してください。\n\n{text}'
)
print(f'モデル: {model}')
print(f'原文: {text}')
print(f'翻訳: {response.text}')

