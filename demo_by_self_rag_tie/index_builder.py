from llama_index import VectorStoreIndex, ServiceContext
from config import RAGConfig
from embedder import get_embed_model
from chunking import get_text_splitter
from qdrant_store import get_qdrant_store
from llama_index.schema import Document
from typing import List


def build_index(documents: List[Document], config: RAGConfig) -> VectorStoreIndex:
    # Initialize embedding model
    embed_model = get_embed_model(config.embedding)
    # Qdrant vector store
    vector_store = get_qdrant_store(config.qdrant)
    
    # Chunking / Text splitting
    text_splitter = get_text_splitter(config.chunking)

    # Service context with embedder and splitter
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        text_splitter=text_splitter
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    
    # Build index with service context and vector store
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        storage_context=storage_context
    )

    return index

def load_existing_index(config: RAGConfig) -> VectorStoreIndex:
    vector_store = get_qdrant_store(config.qdrant)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )