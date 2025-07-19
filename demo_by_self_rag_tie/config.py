from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"

@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50

@dataclass
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "rag_multimodal"

@dataclass
class RetrievalConfig:
    top_k: int = 5

@dataclass
class RAGConfig:
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    qdrant: QdrantConfig = QdrantConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
