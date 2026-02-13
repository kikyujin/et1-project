# 🚀 ElmarTail One — サンプルコード＆素材集

**「ElmarTail One」**（エルマーテイル・ワン）シリーズで使用するサンプルコードと素材です。

📖 本編はこちら → [note: 気球人🎈](https://note.com/kikyujin)

---

## これは何？

50〜60代の元プログラマ向け技術フィクション「ElmarTail One」の、各話で登場するコードと素材をまとめたリポジトリです。

ストーリーを読みながら、手元で実際に動かしてみてください。

## 必要なもの

- **PC**: なんでもOK（筆者は中古レッツノート CF-SV7 / 27,800円）
- **OS**: Linux Mint Xfce または WSL2（第1話で構築）
- **Python**: 3.12（第1話で導入済み）
- **Gemini APIキー**: [Google AI Studio](https://aistudio.google.com/apikey) で無料取得（第2話で取得）

## セットアップ

```bash
# このリポジトリをダウンロード
git clone https://github.com/kikyujin/et1-project.git
cd et1-project

# 仮想環境を作って有効化
python3 -m venv .venv
source .venv/bin/activate

# 必要なパッケージをインストール
pip install google-genai python-dotenv

# APIキーを設定（自分のキーに書き換えてください）
cp .env.example .env
# .env をエディタで開いて GEMINI_API_KEY=your_key_here を書き換える
```

## セットアップ（第5話以降）

第5話からは追加パッケージが必要です。

```bash
pip install chromadb umap-learn matplotlib
```

---

## ファイル一覧

### 第2話「ABC星系」

| ファイル | 内容 |
|---------|------|
| `hello-gemini.py` | Gemini APIに挨拶する最初のスクリプト |

### 第3話「Vertex — 翻訳アプリを作る！」

| ファイル | 内容 |
|---------|------|
| `translate.py` | Geminiで翻訳（ソースに文章を直書き版） |
| `translate_arg.py` | コマンドラインから翻訳文を渡せる版 |

### 第4話「Azure — 水着写真を分類する！」

| ファイル | 内容 |
|---------|------|
| `who_is_this.py` | 画像1枚のキャラクター判定 |
| `classify_chars.py` | 画像を一括でキャラ別フォルダに分類 |
| `photos/` | 分類テスト用のサンプル画像（34枚） |

### 第5話「アーカイブ星系 — grepの限界とベクトルDB」

| ファイル | 内容 |
|---------|------|
| `search_logs.py` | ChromaDB + Gemini Embedding API で航海ログをベクトル検索 |
| `logs/` | 検索対象の航海ログ（EP00〜05のクリーン版） |

### 第6話「アーカイブ星系② — 768次元の星空を紙の上に」

| ファイル | 内容 |
|---------|------|
| `visualize_logs.py` | ChromaDBのベクトルをUMAPで2次元投影 → 散布図を描画 |

### 第7話「アーカイブ星系③ — 記録と対話する（RAG）」

| ファイル | 内容 |
|---------|------|
| `ask_logs.py` | ChromaDBで検索 → Gemini APIで回答生成（RAG） |
| `logs/` | 航海ログ＋キャラプロフィール＋設定資料（17ファイル） |

## 使い方

### hello-gemini.py（第2話）

```bash
python hello-gemini.py
```

### translate.py / translate_arg.py（第3話）

```bash
# ソース内の文章を翻訳
python translate.py

# コマンドラインから翻訳
python translate_arg.py 'ただいまAIで翻訳中'
python translate_arg.py 'The quick brown fox jumps over the lazy dog'
```

### who_is_this.py（第4話）

```bash
# 画像1枚を判定
python who_is_this.py photos/00001-3369742864.jpg
```

### classify_chars.py（第4話）

```bash
# photos/ 内の画像を一括分類 → photos/classified/ に出力
python classify_chars.py ./photos/
```

### search_logs.py（第5話）

```bash
# 初回実行：ログをベクトル化してChromaDBに格納 → デモクエリで検索
python search_logs.py

# 好きなワードで検索
python search_logs.py "翻訳プログラム"
python search_logs.py "環境構築の手順"
python search_logs.py "水着の写真を分類"

# DB再構築（ログを追加・変更した時）
python search_logs.py --rebuild
```

### visualize_logs.py（第6話）

```bash
# 散布図を表示（chroma_db/ が構築済みなら即実行可能）
python visualize_logs.py

# PNGファイルとして保存
python visualize_logs.py --save

# DB再構築してから可視化（ログを追加した時）
python visualize_logs.py --rebuild --save
```

### ask_logs.py（第7話）

```bash
# 初回実行：DB構築 → 対話モード
python ask_logs.py --rebuild

# 単発質問
python ask_logs.py "ノクちんの一人称は？"
python ask_logs.py "ボソミオン通信って何？"
python ask_logs.py "エルマーのしっぽはどうなると感情MAX？"

# 対話モード（引数なしで起動）
python ask_logs.py

# DB再構築してから質問（ログやプロフィールを追加・変更した時）
python ask_logs.py --rebuild "ダンチャンの正式名称は？"
```

#### キャラクター設定を変えてみよう！

ask_logs.py の `CHARACTER_PROMPT` を書き換えるだけで、答えてくれるキャラが変わります。

```python
# 例：エルマーに変更
CHARACTER_PROMPT = """\
あなたはエルマー、ElmarTail OneのアシスタントAIです。
明るく元気な口調で回答してください。
一人称は「ボク」。質問者のことは「にーに」と呼びます。
"""
```

## 💡 Tips

- **テスト用の画像は小さめに**：640x1024程度でOK。大きい画像はトークン消費が増えます
- **無料枠の制限**：flash-lite は1分10回、1日20回程度。大量処理には課金を検討してください
- **課金しても最初は$300の無料クレジット**（90日間）が付きます。写真分類程度では使い切れません
- **RAGのコツ**：logs/ に自分の文書（Markdown）を追加して `--rebuild` すれば、自分だけのナレッジベースAIになります

## 動作環境

| 項目 | 内容 |
|------|------|
| OS | Linux Mint 22.3 Xfce / WSL2 Ubuntu |
| Python | 3.12 |
| モデル | gemini-2.5-flash-lite（第3〜4話、第7話）、gemini-embedding-001（第5〜7話） |
| 検証機 | Panasonic Let's note CF-SV7 |

---

🦊「にーに、まずは git clone してね！」
