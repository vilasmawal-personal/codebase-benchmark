import os
import pickle
from typing import Dict, List


CODE_EXTENSIONS = (".py", ".md", ".rst")


def load_repo(repo_path: str) -> List[Dict[str, str]]:
    documents: List[Dict[str, str]] = []

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(CODE_EXTENSIONS):
                path = os.path.join(root, file)

                try:
                    with open(path, "r", encoding="utf-8") as f:
                        documents.append({
                            "path": path,
                            "content": f.read()
                        })
                except Exception:
                    continue

    return documents


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def build_chunks(
    repo_path: str,
    output_path: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict[str, str]]:
    print("📂 Reading repository...")

    documents = load_repo(repo_path)

    print(f"📄 Loaded {len(documents)} files")

    all_chunks = []

    for doc in documents:
        chunks = chunk_text(
            doc["content"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": doc["path"]
            })

    print(f"🧩 Created {len(all_chunks)} chunks")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"✅ Saved chunks → {output_path}")

    return all_chunks
