import os
import pickle


CODE_EXTENSIONS = (".py", ".md", ".rst")


def load_repo(repo_path):
    documents = []

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


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def build_chunks(repo_path, output_path):
    print("📂 Reading repository...")

    documents = load_repo(repo_path)

    print(f"📄 Loaded {len(documents)} files")

    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["content"])

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