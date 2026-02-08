"""
ElmarTail One - Episode 4: この写真は誰？
Gemini flash-lite に画像を見せて、キャラクターを聞くだけのシンプル版

使い方:
  python who_is_this.py <画像ファイル>
  python who_is_this.py ./photos/00001.jpg
"""

import os
import sys
import base64
from pathlib import Path
from dotenv import load_dotenv
from google import genai

MODEL = "gemini-2.5-flash-lite"

CHARACTER_PROFILES = """
以下のキャラクターから、画像に写っている人物を判定してください。

1. エルマー — 金髪、青い瞳、狐耳・狐しっぽ、女性
2. ノクちん — 黒髪ウェーブ、ブラウンの瞳、黒い水着、小柄、女性
3. スミレん — スミレ色（紫系）ショートボブ、眼鏡、女性
4. マスター — 男性、短い黒髪、30代

キャラ名だけ答えてください。
"""


def main():
    if len(sys.argv) < 2:
        print("使い方: python who_is_this.py <画像ファイル>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        print(f"エラー: '{image_path}' が見つかりません")
        sys.exit(1)

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("エラー: GEMINI_API_KEY が .env に設定されていません")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # 画像読み込み
    ext = image_path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # Geminiに聞く
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": CHARACTER_PROFILES},
                    {"inline_data": {"mime_type": mime_type, "data": image_data}},
                    {"text": "この写真は誰？"},
                ],
            }
        ],
    )

    print(response.text.strip())


if __name__ == "__main__":
    main()
