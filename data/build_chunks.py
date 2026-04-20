import ast
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


def _split_by_lines(text: str, max_lines: int) -> List[str]:
    lines = text.splitlines()
    if not lines:
        return []

    return [
        "\n".join(lines[i:i + max_lines])
        for i in range(0, len(lines), max_lines)
    ]


def chunk_python_code(
    code: str,
    source_path: str,
    max_lines: int = 30,
) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                chunk_text = ast.get_source_segment(code, node)
                if not chunk_text:
                    continue

                line_chunks = _split_by_lines(chunk_text, max_lines)
                if len(line_chunks) > 1:
                    for i, sub_chunk in enumerate(line_chunks, start=1):
                        chunks.append({
                            "text": sub_chunk,
                            "source": f"{source_path}:{node.name}_part{i}"
                        })
                else:
                    chunks.append({
                        "text": chunk_text,
                        "source": f"{source_path}:{node.name}"
                    })

            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                chunk_text = ast.get_source_segment(code, node)
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "source": f"{source_path}:variable"
                    })

    except Exception:
        # Fallback: plain line-based chunking.
        for i, sub_chunk in enumerate(_split_by_lines(code, max_lines), start=1):
            chunks.append({
                "text": sub_chunk,
                "source": f"{source_path}:fallback_part{i}"
            })

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
        if doc["path"].endswith(".py"):
            py_chunks = chunk_python_code(
                doc["content"],
                source_path=doc["path"],
            )

            if py_chunks:
                all_chunks.extend(py_chunks)
                continue

        chunks = chunk_text(doc["content"], chunk_size=chunk_size, overlap=overlap)
        for chunk in chunks:
            all_chunks.append({"text": chunk, "source": doc["path"]})

    print(f"🧩 Created {len(all_chunks)} chunks")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"✅ Saved chunks → {output_path}")

    return all_chunks
