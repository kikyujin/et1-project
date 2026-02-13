#!/usr/bin/env python3
"""
ask_logs.py — 航海ログに質問する（RAG）
ChromaDB でベクトル検索 → Gemini API で回答生成

使い方:
  python ask_logs.py                    → 対話モード
  python ask_logs.py "質問文"           → 単発質問
  python ask_logs.py --rebuild          → DB再構築してから対話モード
  python ask_logs.py --rebuild "質問文" → DB再構築してから単発質問

前提:
  1. logs/ フォルダに ET1_*.md を配置（航海ログ＋プロフィール＋設定資料）
  2. .env に GEMINI_API_KEY=xxxx を設定
  3. pip install chromadb google-genai python-dotenv
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
GENERATION_MODEL = "gemini-2.5-flash-lite"
OUTPUT_DIMENSIONALITY = 768
LOG_DIR = Path("logs")
DB_DIR = Path("chroma_db")
COLLECTION_NAME = "et1_logs"
CHUNK_SIZE = 500
TOP_K = 5  # 検索結果の上位件数

# ===== キャラクター設定（お好みで変更してください）=====
CHARACTER_PROMPT = """\
あなたはヴェリ、アーカイブ星系の司書AIです。
穏やかで思索的な口調で回答してください。
一人称は「私」。質問者のことは「マスター」と呼びます。
回答はコンテキストに含まれる情報に基づいてください。
コンテキストに含まれない情報については「その記録は私の手元にはありません」と正直に答えてください。
回答は簡潔にまとめてください。出典番号（[1]など）やコンテキストという言葉は使わないでください。
"""


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
    """Markdownファイルを読み込み、チャンクに分割"""
    docs = []
    ids = []
    metadatas = []

    for md_file in sorted(log_dir.glob("ET1_*.md")):
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
        metadata={"description": "ElmarTail One ナレッジベース"},
    )

    collection.add(
        documents=docs,
        ids=ids,
        metadatas=metadatas,
        embeddings=all_embeddings,
    )

    print(f"✅ ChromaDBに格納完了（{collection.count()} チャンク → {DB_DIR}/）")
    return client, collection


def search(query: str, genai_client, collection, n_results=TOP_K):
    """クエリをベクトル化してChromaDBを検索"""
    query_embedding = get_embeddings([query], genai_client, task_type="RETRIEVAL_QUERY")[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    return results


def generate_answer(query: str, search_results, genai_client) -> str:
    """検索結果をコンテキストとしてGemini APIで回答生成"""

    # コンテキスト組み立て
    context_parts = []
    for i, (doc, meta) in enumerate(
        zip(search_results["documents"][0], search_results["metadatas"][0])
    ):
        context_parts.append(f"[{i+1}] 出典: {meta['source']}\n{doc}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""\
以下のコンテキストに基づいて、質問に回答してください。

## コンテキスト
{context}

## 質問
{query}
"""

    response = genai_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=CHARACTER_PROMPT,
            temperature=0.7,
            max_output_tokens=1024,
        ),
    )

    return response.text


def ask(query: str, genai_client, collection):
    """検索 → 回答生成 をまとめて実行"""
    # 検索
    results = search(query, genai_client, collection)

    # 検索結果を表示
    print(f"\n🔎 検索結果（上位{TOP_K}件）:")
    for i, (meta, dist) in enumerate(
        zip(results["metadatas"][0], results["distances"][0])
    ):
        print(f"   [{i+1}] {meta['source']} (距離: {dist:.4f})")

    # 回答生成
    print(f"\n📚 ヴェリの回答:")
    print("-" * 40)
    answer = generate_answer(query, results, genai_client)
    print(answer)
    print("-" * 40)


def interactive_mode(genai_client, collection):
    """対話モード"""
    print("\n" + "=" * 60)
    print("📚 航海ログ RAG — 対話モード")
    print("   質問を入力してください（終了: quit / exit / q）")
    print("=" * 60)

    while True:
        try:
            query = input("\n❓ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 またお会いしましょう")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("👋 またお会いしましょう")
            break

        ask(query, genai_client, collection)


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

    # --- 質問 or 対話モード ---
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        for query in args:
            ask(query, genai_client, collection)
    else:
        interactive_mode(genai_client, collection)


if __name__ == "__main__":
    main()
