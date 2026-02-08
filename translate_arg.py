# translate.py（改良版）
from google import genai
import os, sys
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

model = 'gemini-2.5-flash-lite'
text = ' '.join(sys.argv[1:])

if not text:
    print('使い方: python translate.py 翻訳したい文章')
    sys.exit(1)

response = client.models.generate_content(
    model=model,
    contents=f'以下の文章を、日本語なら英語に、英語なら日本語に翻訳してください。翻訳文のみを返してください。\n\n{text}'
)
print(response.text)
