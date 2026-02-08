"""
ElmarTail One - Episode 4: キャラクター画像分類スクリプト
Gemini 2.5 flash-lite の Vision API でキャラクターを判定し、フォルダ分けする

使い方:
  python classify_chars.py <画像ディレクトリ>
  python classify_chars.py ./photos
"""

import os
import sys
import shutil
import base64
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError

# --- 設定 ---
MODEL = "gemini-2.5-flash-lite"

# キャラクター定義（Geminiに渡す判定基準）
CHARACTER_PROFILES = """
以下のキャラクターから、画像に写っている人物を判定してください。

## キャラクター一覧

1. エルマー — 金髪、青い瞳、狐耳・狐しっぽ、女性
2. ノクちん — 黒髪ウェーブ、ブラウンの瞳、黒い水着、小柄、女性
3. スミレん — スミレ色（紫系）ショートボブ、眼鏡、女性
4. マスター — 男性、短い黒髪、30代

## ルール
- 必ず上記4名のうち1名を選んでください
- 回答はキャラクターの英語ID（elmar, nokuchin, sumiren, master）のみを返してください
- 余計な説明は不要です
"""

# 対応する画像拡張子
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def load_image_as_base64(image_path: str) -> tuple[str, str]:
    """画像をBase64エンコードして返す"""
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime_type = mime_map.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    return data, mime_type


def classify_image(client: genai.Client, image_path: str) -> str:
    """Gemini Vision APIで画像のキャラクターを判定（レート制限対応）"""
    image_data, mime_type = load_image_as_base64(image_path)

    for attempt in range(3):  # 最大3回リトライ
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": CHARACTER_PROFILES},
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": image_data,
                                }
                            },
                            {"text": "この画像のキャラクターは誰ですか？英語IDのみで答えてください。"},
                        ],
                    }
                ],
            )
            break
        except ClientError as e:
            if "429" in str(e):
                wait = 30 * (attempt + 1)
                print(f"\n  ⏳ レート制限！ {wait}秒待機中...", end="", flush=True)
                time.sleep(wait)
                print(" リトライ → ", end="", flush=True)
            else:
                raise

    # レスポンスからキャラ名を取得（余計な空白・改行を除去）
    result = response.text.strip().lower()

    # 有効なキャラ名かチェック
    valid_ids = {"elmar", "nokuchin", "sumiren", "master"}
    if result not in valid_ids:
        print(f"  ⚠ 判定結果が想定外: '{result}' → unknown に分類")
        return "unknown"

    return result


def main():
    # 引数チェック
    if len(sys.argv) < 2:
        print("使い方: python classify_chars.py <画像ディレクトリ>")
        sys.exit(1)

    source_dir = Path(sys.argv[1])
    if not source_dir.is_dir():
        print(f"エラー: '{source_dir}' はディレクトリではありません")
        sys.exit(1)

    # .envからAPIキー読み込み
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("エラー: GEMINI_API_KEY が .env に設定されていません")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # 出力ディレクトリのベース
    output_base = source_dir / "classified"

    # 画像ファイルを収集（macOSのリソースファイル ._xxx を除外）
    images = [
        f for f in sorted(source_dir.iterdir())
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        and not f.name.startswith("._")
    ]

    if not images:
        print(f"'{source_dir}' に画像ファイルが見つかりません")
        sys.exit(1)

    print(f"🚀 {len(images)} 枚の画像を分類します")
    print(f"   モデル: {MODEL}")
    print(f"   出力先: {output_base}/")
    print()

    # 分類結果の集計
    results = {}

    for image_path in images:
        print(f"📷 {image_path.name} → ", end="", flush=True)

        char_id = classify_image(client, str(image_path))
        print(f"{'🦊' if char_id == 'elmar' else '🖤' if char_id == 'nokuchin' else '🪷' if char_id == 'sumiren' else '👨' if char_id == 'master' else '❓'} {char_id}")

        # フォルダ作成＆コピー
        char_dir = output_base / char_id
        char_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, char_dir / image_path.name)

        # 集計
        results[char_id] = results.get(char_id, 0) + 1

    # 結果サマリー
    print()
    print("=" * 40)
    print("📊 分類結果:")
    for char_id, count in sorted(results.items()):
        emoji = {'elmar': '🦊', 'nokuchin': '🖤', 'sumiren': '🪷', 'master': '👨'}.get(char_id, '❓')
        print(f"  {emoji} {char_id}: {count} 枚")
    print(f"  合計: {sum(results.values())} 枚")
    print(f"  出力先: {output_base}/")


if __name__ == "__main__":
    main()
