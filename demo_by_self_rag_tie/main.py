from rag_component import MultimodalRAG
from config import RAGConfig

# Optional: customize config
custom_config = RAGConfig()
custom_config.embedding.model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
custom_config.chunking.chunk_size = 256
custom_config.qdrant.collection_name = "my_custom_collection"
custom_config.retrieval.top_k = 3

# Initialize component
rag = MultimodalRAG(config=custom_config)

# Load and index documents
rag.load_and_index(["docs/myfile.pdf", "images/diagram.png"])

# Ask a question
response = rag.query("What is shown in the diagram?")
print(response)
