from llama_index.retrievers import VectorIndexRetriever
from llama_index.indices.base import BaseIndex
from config import RetrievalConfig

def get_retriever(index: BaseIndex, config: RetrievalConfig) -> VectorIndexRetriever:
    return VectorIndexRetriever(
        index=index,
        similarity_top_k=config.top_k
    )
