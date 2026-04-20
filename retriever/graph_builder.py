import os
from typing import List, Dict, Tuple
from tree_sitter_languages import get_parser


# Supported extensions (extend later if needed)
CODE_EXTENSIONS = (".py",)


class ASTGraphBuilder:
    """
    Build a graph from codebase using Tree-sitter AST.

    Extracts:
    - Nodes: file, class, function
    - Edges: contains, calls, imports
    """

    def __init__(self, language: str = "python"):
        self.parser = get_parser(language)

        self.nodes: List[Dict] = []
        self.edges: List[Dict] = []

        self.node_id_counter = 0

    # --------------------------------------------------
    # 🔹 Public API
    # --------------------------------------------------
    def build_from_repo(self, repo_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Build graph from repository

        Returns:
            nodes, edges
        """

        print("📂 Building AST graph...")

        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(CODE_EXTENSIONS):
                    full_path = os.path.join(root, file)
                    self._parse_file(full_path)

        print(f"🧩 Nodes: {len(self.nodes)}")
        print(f"🔗 Edges: {len(self.edges)}")

        return self.nodes, self.edges

    # --------------------------------------------------
    # 🔹 File parsing
    # --------------------------------------------------
    def _parse_file(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()
        except Exception:
            return

        tree = self.parser.parse(bytes(code, "utf8"))
        root = tree.root_node

        file_id = self._add_node(
            node_type="file",
            name=os.path.basename(path),
            text=code,
            file=path
        )

        self._traverse(root, code, path, parent_id=file_id)

    # --------------------------------------------------
    # 🔹 AST traversal
    # --------------------------------------------------
    def _traverse(self, node, code: str, path: str, parent_id: int):
        """
        Recursively traverse AST
        """

        # ----------------------------
        # Class
        # ----------------------------
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = self._get_text(name_node, code)

                class_id = self._add_node(
                    node_type="class",
                    name=name,
                    text=self._get_text(node, code),
                    file=path
                )

                self._add_edge(parent_id, class_id, "contains")
                parent_id = class_id

        # ----------------------------
        # Function
        # ----------------------------
        elif node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = self._get_text(name_node, code)

                func_id = self._add_node(
                    node_type="function",
                    name=name,
                    text=self._get_text(node, code),
                    file=path
                )

                self._add_edge(parent_id, func_id, "contains")
                parent_id = func_id

        # ----------------------------
        # Import
        # ----------------------------
        elif node.type in ("import_statement", "import_from_statement"):
            import_text = self._get_text(node, code)

            import_id = self._add_node(
                node_type="import",
                name=import_text,
                text=import_text,
                file=path
            )

            self._add_edge(parent_id, import_id, "imports")

        # ----------------------------
        # Function Call (basic)
        # ----------------------------
        elif node.type == "call":
            func_node = node.child_by_field_name("function")

            if func_node:
                func_name = self._get_text(func_node, code)

                call_id = self._add_node(
                    node_type="call",
                    name=func_name,
                    text=func_name,
                    file=path
                )

                self._add_edge(parent_id, call_id, "calls")

        # ----------------------------
        # Traverse children
        # ----------------------------
        for child in node.children:
            self._traverse(child, code, path, parent_id)

    # --------------------------------------------------
    # 🔹 Node / Edge helpers
    # --------------------------------------------------
    def _add_node(self, node_type: str, name: str, text: str, file: str) -> int:
        node_id = self.node_id_counter
        self.node_id_counter += 1

        self.nodes.append({
            "id": node_id,
            "type": node_type,
            "name": name,
            "text": text,
            "file": file
        })

        return node_id

    def _add_edge(self, src: int, dst: int, relation: str):
        self.edges.append({
            "source": src,
            "target": dst,
            "type": relation
        })

    # --------------------------------------------------
    # 🔹 Utility
    # --------------------------------------------------
    def _get_text(self, node, code: str) -> str:
        return code[node.start_byte:node.end_byte]