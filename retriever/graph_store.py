from typing import List, Dict, Any, Optional
import os
import json
import kuzu


class GraphStore:
    """
    Graph storage using Kùzu DB.

    Stores:
    - Nodes (file, class, function, etc.)
    - Edges (contains, calls, imports)
    - Embeddings (as JSON arrays)

    Provides:
    - Insert
    - Query
    - Neighbor lookup
    """

    def __init__(self, db_path: str = "graph_db"):
        """
        Args:
            db_path: directory for Kùzu database
        """

        os.makedirs(db_path, exist_ok=True)

        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)

        self._create_schema()

    # --------------------------------------------------
    # 🔹 Schema
    # --------------------------------------------------
    def _create_schema(self):
        """
        Create tables if not exist
        """

        # Nodes
        self.conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Node(
            id INT64,
            name STRING,
            type STRING,
            file STRING,
            text STRING,
            embedding STRING,
            PRIMARY KEY(id)
        )
        """)

        # Edges
        self.conn.execute("""
        CREATE REL TABLE IF NOT EXISTS Rel(
            FROM Node TO Node,
            type STRING
        )
        """)

    # --------------------------------------------------
    # 🔹 Insert Nodes
    # --------------------------------------------------
    def insert_nodes(self, nodes: List[Dict]):
        """
        Insert nodes into DB
        """

        print(f"📥 Inserting {len(nodes)} nodes...")

        for node in nodes:
            embedding = node.get("embedding", None)

            # Convert embedding to JSON string
            emb_str = json.dumps(embedding.tolist()) if embedding is not None else None

            self.conn.execute(
                """
                INSERT INTO Node VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    node["id"],
                    node.get("name", ""),
                    node.get("type", ""),
                    node.get("file", ""),
                    node.get("text", ""),
                    emb_str,
                ],
            )

    # --------------------------------------------------
    # 🔹 Insert Edges
    # --------------------------------------------------
    def insert_edges(self, edges: List[Dict]):
        """
        Insert edges into DB
        """

        print(f"🔗 Inserting {len(edges)} edges...")

        for e in edges:
            self.conn.execute(
                """
                MATCH (a:Node), (b:Node)
                WHERE a.id = ? AND b.id = ?
                CREATE (a)-[:Rel {type: ?}]->(b)
                """,
                [
                    e["source"],
                    e["target"],
                    e["type"],
                ],
            )

    # --------------------------------------------------
    # 🔹 Bulk insert (recommended)
    # --------------------------------------------------
    def insert_graph(self, nodes: List[Dict], edges: List[Dict]):
        """
        Insert full graph
        """

        self.insert_nodes(nodes)
        self.insert_edges(edges)

    # --------------------------------------------------
    # 🔹 Get node by id
    # --------------------------------------------------
    def get_node(self, node_id: int) -> Optional[Dict]:
        result = self.conn.execute(
            "MATCH (n:Node) WHERE n.id = ? RETURN n",
            [node_id],
        ).get_all()

        if not result:
            return None

        return self._parse_node(result[0][0])

    # --------------------------------------------------
    # 🔹 Get neighbors
    # --------------------------------------------------
    def get_neighbors(self, node_id: int) -> List[int]:
        result = self.conn.execute(
            """
            MATCH (a:Node)-[r:Rel]->(b:Node)
            WHERE a.id = ?
            RETURN b.id
            """,
            [node_id],
        ).get_all()

        return [row[0] for row in result]

    # --------------------------------------------------
    # 🔹 Get nodes by type
    # --------------------------------------------------
    def get_nodes_by_type(self, node_type: str) -> List[Dict]:
        result = self.conn.execute(
            """
            MATCH (n:Node)
            WHERE n.type = ?
            RETURN n
            """,
            [node_type],
        ).get_all()

        return [self._parse_node(row[0]) for row in result]

    # --------------------------------------------------
    # 🔹 Load all nodes (for embedding retrieval)
    # --------------------------------------------------
    def load_all_nodes(self) -> List[Dict]:
        result = self.conn.execute(
            "MATCH (n:Node) RETURN n"
        ).get_all()

        return [self._parse_node(row[0]) for row in result]

    # --------------------------------------------------
    # 🔹 Parse node
    # --------------------------------------------------
    def _parse_node(self, node_obj: Any) -> Dict:
        """
        Convert Kùzu node → Python dict
        """

        emb = node_obj["embedding"]
        embedding = json.loads(emb) if emb else None

        return {
            "id": node_obj["id"],
            "name": node_obj["name"],
            "type": node_obj["type"],
            "file": node_obj["file"],
            "text": node_obj["text"],
            "embedding": embedding,
        }

    # --------------------------------------------------
    # 🔹 Clear DB
    # --------------------------------------------------
    def clear(self):
        self.conn.execute("MATCH (n:Node) DELETE n")

    def __repr__(self):
        return "GraphStore(Kùzu)"