from llama_index.text_splitter import SentenceSplitter
from config import ChunkingConfig

def get_text_splitter(config: ChunkingConfig):
    return SentenceSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
