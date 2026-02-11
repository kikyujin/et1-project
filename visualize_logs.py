#!/usr/bin/env python3
"""
visualize_logs.py — ChromaDBのベクトルをUMAPで2次元に投影して散布図を描画する

前提:
  - search_logs.py で chroma_db/ が構築済みであること
  - pip install umap-learn matplotlib

使い方:
  python visualize_logs.py                → 散布図を表示
  python visualize_logs.py --save         → umap_logs.png として保存
  python visualize_logs.py --rebuild      → DB再構築してから可視化
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb
import numpy as np

load_dotenv()

# --- 設定（search_logs.py と共通）---
EMBEDDING_MODEL = "gemini-embedding-001"
OUTPUT_DIMENSIONALITY = 768
LOG_DIR = Path("logs")
DB_DIR = Path("chroma_db")
COLLECTION_NAME = "et1_logs"
CHUNK_SIZE = 500

# --- エピソード別の表示設定 ---
EPISODE_COLORS = {
    "ET1_ep00": "#888888",   # 覚醒（グレー）
    "ET1_ep01": "#4CAF50",   # Linux セットアップ（緑）
    "ET1_ep01x": "#81C784",  # 外伝（薄緑）
    "ET1_ep02": "#2196F3",   # ABC星系・APIキー（青）
    "ET1_ep03": "#FF9800",   # 翻訳プログラム（オレンジ）
    "ET1_ep04": "#E91E63",   # 水着回・Vision（ピンク）
    "ET1_ep05": "#9C27B0",   # アーカイブ・ベクトルDB（紫）
}

EPISODE_LABELS = {
    "ET1_ep00": "EP00: 覚醒",
    "ET1_ep01": "EP01: Linux セットアップ",
    "ET1_ep01x": "EP01x: 環境構築外伝",
    "ET1_ep02": "EP02: APIキー取得",
    "ET1_ep03": "EP03: 翻訳プログラム",
    "ET1_ep04": "EP04: Vision・画像分類",
    "ET1_ep05": "EP05: ベクトルDB",
}


# --- search_logs.py から流用 ---
def get_embeddings(texts, client, task_type="RETRIEVAL_DOCUMENT"):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=OUTPUT_DIMENSIONALITY,
        ),
    )
    return [e.values for e in result.embeddings]


def load_and_chunk(log_dir):
    docs, ids, metadatas = [], [], []
    for md_file in sorted(log_dir.glob("ET1_ep*.md")):
        text = md_file.read_text(encoding="utf-8")
        paragraphs = text.split("\n\n")
        chunk = ""
        chunk_idx = 0
        for para in paragraphs:
            if len(chunk) + len(para) < CHUNK_SIZE:
                chunk += para + "\n\n"
            else:
                if chunk.strip():
                    doc_id = f"{md_file.stem}_chunk{chunk_idx:03d}"
                    docs.append(chunk.strip())
                    ids.append(doc_id)
                    metadatas.append({
                        "source": md_file.name,
                        "episode": md_file.stem,
                        "chunk_index": chunk_idx,
                    })
                    chunk_idx += 1
                chunk = para + "\n\n"
        if chunk.strip():
            doc_id = f"{md_file.stem}_chunk{chunk_idx:03d}"
            docs.append(chunk.strip())
            ids.append(doc_id)
            metadatas.append({
                "source": md_file.name,
                "episode": md_file.stem,
                "chunk_index": chunk_idx,
            })
    return docs, ids, metadatas


def build_db(docs, ids, metadatas, genai_client):
    print(f"📦 {len(docs)} チャンクを読み込みました")
    print(f"🔄 {EMBEDDING_MODEL} でベクトル化中（{OUTPUT_DIMENSIONALITY}次元）...")
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        embeddings = get_embeddings(batch, genai_client, task_type="RETRIEVAL_DOCUMENT")
        all_embeddings.extend(embeddings)
        print(f"   {min(i + batch_size, len(docs))}/{len(docs)} 完了")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "ElmarTail One 航海ログ"},
    )
    collection.add(
        documents=docs, ids=ids,
        metadatas=metadatas, embeddings=all_embeddings,
    )
    print(f"✅ ChromaDBに格納完了（{collection.count()} チャンク → {DB_DIR}/）")
    return client, collection


# --- UMAP可視化 ---
def visualize(collection, save=False):
    import umap
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 日本語フォント設定（japanize_matplotlib不要）
    # macOS: ヒラギノ、Linux: Noto Sans CJK、Windows: Yu Gothic
    jp_fonts = ["Hiragino Sans", "Hiragino Maru Gothic Pro",
                "Noto Sans CJK JP", "Noto Sans JP",
                "Yu Gothic", "Meiryo", "IPAexGothic"]
    for font_name in jp_fonts:
        try:
            font_manager.findfont(font_name, fallback_to_default=False)
            matplotlib.rcParams["font.family"] = font_name
            break
        except ValueError:
            continue

    # ChromaDBから全ベクトルとメタデータを取得
    all_data = collection.get(include=["embeddings", "metadatas"])
    embeddings = np.array(all_data["embeddings"])
    metadatas = all_data["metadatas"]

    print(f"🔄 UMAP実行中... {embeddings.shape[0]} チャンク × {embeddings.shape[1]} 次元 → 2次元")

    # UMAP で2次元に投影
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=10,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)

    print("✅ UMAP完了")

    # --- 散布図を描画 ---
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # エピソード別にプロット
    episodes = sorted(set(m["episode"] for m in metadatas))
    for ep in episodes:
        mask = [m["episode"] == ep for m in metadatas]
        x = coords[mask, 0]
        y = coords[mask, 1]
        color = EPISODE_COLORS.get(ep, "#FFFFFF")
        label = EPISODE_LABELS.get(ep, ep)
        ax.scatter(x, y, c=color, label=label, s=60, alpha=0.8, edgecolors="white", linewidth=0.3)

    ax.set_title("ElmarTail One 航海ログ — 768次元の星空を2次元に", fontsize=16, color="white", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = ax.legend(
        loc="upper right", fontsize=10,
        facecolor="#2a2a4a", edgecolor="#555",
        labelcolor="white",
        bbox_to_anchor=(1.0, 1.0),
    )
    legend.get_frame().set_alpha(0.9)

    plt.tight_layout()

    if save:
        output_path = "umap_logs.png"
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"💾 保存しました: {output_path}")
    else:
        plt.show()


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY が設定されていません（.env を確認）")
        sys.exit(1)

    genai_client = genai.Client(api_key=api_key)

    # DB構築 or 読み込み
    if not DB_DIR.exists() or "--rebuild" in sys.argv:
        docs, ids, metadatas = load_and_chunk(LOG_DIR)
        if not docs:
            print(f"❌ {LOG_DIR}/ にログファイルがありません")
            sys.exit(1)
        db_client, collection = build_db(docs, ids, metadatas, genai_client)
    else:
        print(f"📂 既存DB読み込み: {DB_DIR}/")
        db_client = chromadb.PersistentClient(path=str(DB_DIR))
        collection = db_client.get_collection(COLLECTION_NAME)
        print(f"   {collection.count()} チャンク")

    # 可視化
    save = "--save" in sys.argv
    visualize(collection, save=save)


if __name__ == "__main__":
    main()
