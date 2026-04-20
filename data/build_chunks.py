import ast
import os
import pickle
from typing import Dict, List, Optional

from tree_sitter_languages import get_parser


TEXT_EXTENSIONS = (".md", ".rst", ".txt")
CODE_EXTENSIONS = (
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".go", ".rs", ".c", ".h", ".cpp", ".hpp",
    ".cs", ".php", ".rb", ".swift", ".kt", ".kts",
    ".scala", ".sql", ".sh", ".yaml", ".yml", ".json",
)
ALLOWED_EXTENSIONS = CODE_EXTENSIONS + TEXT_EXTENSIONS


TREE_SITTER_LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
}


def load_repo(repo_path: str) -> List[Dict[str, str]]:
    documents: List[Dict[str, str]] = []

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(ALLOWED_EXTENSIONS):
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


def _safe_get_parser(language: str):
    try:
        return get_parser(language)
    except Exception:
        return None


def _extract_symbol_name(node, code: str) -> Optional[str]:
    name_node = node.child_by_field_name("name")
    if name_node:
        return code[name_node.start_byte:name_node.end_byte].strip()

    # Best effort fallback for languages that do not expose field names uniformly.
    for child in node.children:
        if child.type == "identifier":
            return code[child.start_byte:child.end_byte].strip()

    return None


def _split_large_snippet(
    snippet: str,
    source_path: str,
    label: str,
    max_lines: int,
) -> List[Dict[str, str]]:
    line_chunks = _split_by_lines(snippet, max_lines)
    if len(line_chunks) <= 1:
        return [{"text": snippet, "source": f"{source_path}:{label}"}]

    return [
        {"text": sub_chunk, "source": f"{source_path}:{label}_part{i}"}
        for i, sub_chunk in enumerate(line_chunks, start=1)
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


def chunk_code_with_tree_sitter(
    code: str,
    source_path: str,
    extension: str,
    max_lines: int = 30,
) -> List[Dict[str, str]]:
    language = TREE_SITTER_LANGUAGE_BY_EXTENSION.get(extension)
    if not language:
        return []

    parser = _safe_get_parser(language)
    if parser is None:
        return []

    definition_like_nodes = {
        "function_definition",
        "method_definition",
        "class_definition",
        "interface_declaration",
        "function_declaration",
        "method_declaration",
        "constructor_declaration",
        "struct_item",
        "trait_item",
        "impl_item",
        "enum_item",
    }

    chunks: List[Dict[str, str]] = []
    tree = parser.parse(bytes(code, "utf8"))
    stack = [tree.root_node]

    while stack:
        node = stack.pop()
        stack.extend(node.children)

        if node.type not in definition_like_nodes:
            continue

        snippet = code[node.start_byte:node.end_byte]
        if not snippet.strip():
            continue

        symbol_name = _extract_symbol_name(node, code)
        label = symbol_name if symbol_name else node.type
        chunks.extend(_split_large_snippet(snippet, source_path, label, max_lines))

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

        _, ext = os.path.splitext(doc["path"])
        ast_chunks = chunk_code_with_tree_sitter(
            doc["content"],
            source_path=doc["path"],
            extension=ext.lower(),
        )
        if ast_chunks:
            all_chunks.extend(ast_chunks)
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
