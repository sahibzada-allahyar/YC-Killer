from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any
import json
from pathlib import Path

@dataclass
class Node:
    key: str
    role: str
    content: str = ""
    children: List[str] = field(default_factory=list)

@dataclass
class PaperGraph:
    nodes: Dict[str, Node] = field(default_factory=dict)
    root: str = "title"

    def add(self, node: Node):
        self.nodes[node.key] = node

    def link(self, parent: str, child: str):
        self.nodes[parent].children.append(child)

    def to_json(self, path: Path):
        path.write_text(json.dumps({k: {"role": n.role, "content": n.content, "children": n.children}
                                    for k,n in self.nodes.items()}, indent=2), encoding="utf-8")
