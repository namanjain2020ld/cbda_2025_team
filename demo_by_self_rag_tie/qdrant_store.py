from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from config import QdrantConfig

def get_qdrant_store(config: QdrantConfig):
    client = QdrantClient(
        host=config.host,
        port=config.port
    )
    return QdrantVectorStore(
        client=client,
        collection_name=config.collection_name
    )
