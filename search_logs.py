#!/usr/bin/env python3
"""
search_logs.py — ChromaDB + Gemini Embedding API で航海ログを検索する

使い方:
  1. logs/ フォルダに ET1_ep*.md を配置
  2. .env に GEMINI_API_KEY=xxxx を設定
  3. python search_logs.py                    → デモクエリで検索
     python search_logs.py "検索ワード"       → 指定ワードで検索
     python search_logs.py --rebuild          → DB再構築

埋め込みモデル: gemini-embedding-001
  - text-embedding-004 は 2026-01-14 に非推奨
  - 3072次元（デフォルト）、768次元に縮小してストレージ節約
  - task_type: RETRIEVAL_DOCUMENT（格納時）/ RETRIEVAL_QUERY（検索時）
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb

load_dotenv()

# --- 設定 ---
EMBEDDING_MODEL = "gemini-embedding-001"
OUTPUT_DIMENSIONALITY = 768  # 3072がフル。768で十分実用的＆軽量
LOG_DIR = Path("logs")
DB_DIR = Path("chroma_db")
COLLECTION_NAME = "et1_logs"
CHUNK_SIZE = 500  # チャンクの目安文字数


def get_embeddings(
    texts: list[str],
    client,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """Gemini Embedding API でテキストをベクトル化"""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=OUTPUT_DIMENSIONALITY,
        ),
    )
    return [e.values for e in result.embeddings]


def load_and_chunk(log_dir: Path) -> tuple[list[str], list[str], list[dict]]:
    """ログファイルを読み込み、チャンクに分割"""
    docs = []
    ids = []
    metadatas = []

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
    """ChromaDBにベクトルを格納（永続化）"""
    print(f"📦 {len(docs)} チャンクを読み込みました")

    # 埋め込み生成（バッチ、最大100件ずつ）
    # ドキュメント格納時は RETRIEVAL_DOCUMENT
    print(f"🔄 {EMBEDDING_MODEL} でベクトル化中（{OUTPUT_DIMENSIONALITY}次元）...")
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        embeddings = get_embeddings(batch, genai_client, task_type="RETRIEVAL_DOCUMENT")
        all_embeddings.extend(embeddings)
        print(f"   {min(i + batch_size, len(docs))}/{len(docs)} 完了")

    # ChromaDB に格納
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # 既存コレクションがあれば削除して再作成
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "ElmarTail One 航海ログ"},
    )

    collection.add(
        documents=docs,
        ids=ids,
        metadatas=metadatas,
        embeddings=all_embeddings,
    )

    print(f"✅ ChromaDBに格納完了（{collection.count()} チャンク → {DB_DIR}/）")
    return client, collection


def search(query: str, genai_client, collection, n_results=3):
    """クエリをベクトル化してChromaDBを検索"""
    # 検索時は RETRIEVAL_QUERY
    query_embedding = get_embeddings([query], genai_client, task_type="RETRIEVAL_QUERY")[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    print(f"\n🔎 「{query}」")
    for i, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):
        preview = doc[:120].replace("\n", " ")
        print(f"   [{i + 1}] {meta['source']} (距離: {dist:.4f})")
        print(f"       {preview}...")


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY が設定されていません（.env を確認）")
        sys.exit(1)

    genai_client = genai.Client(api_key=api_key)

    # --- DB構築 ---
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

    # --- 検索 ---
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        for query in args:
            search(query, genai_client, collection)
    else:
        # デモクエリ
        demo_queries = [
            "翻訳プログラム",
            "環境構築の手順",
            "APIキーの取得方法",
            "水着の写真を分類",
            "コールドスリープから目覚めた",
            "ダイソン球",
            "ノクちんとの出会い",
            "VB6やDelphiを使っていた頃",
        ]
        print("\n" + "=" * 60)
        print("🔍 ベクトル検索デモ")
        print("=" * 60)
        for q in demo_queries:
            search(q, genai_client, collection)


if __name__ == "__main__":
    main()
