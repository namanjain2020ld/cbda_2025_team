from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import EmbeddingConfig

def get_embed_model(config: EmbeddingConfig):
    return HuggingFaceEmbedding(
        model_name=config.model_name,
        device=config.device
    )
