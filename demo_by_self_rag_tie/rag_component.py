from config import RAGConfig
from loader import load_documents
from index_builder import build_index, load_existing_index
from retriever import get_retriever
from llama_index import ResponseSynthesizer
from llama_index.query_engine import RetrieverQueryEngine
from typing import List, Optional


class MultimodalRAG:
    def __init__(self, config: Optional[RAGConfig] = None):
        # Initialize the configuration, or use default
        self.config = config or RAGConfig()

        # These will be set later in the pipeline
        self.index = None
        self.retriever = None
        self.query_engine = None

    def load_and_index(self, file_paths: List[str], force_rebuild: bool = False):
        """
        Load files and build or load index.
        If force_rebuild is True, it will re-index from scratch and push to Qdrant.
        Otherwise, it tries to load the existing index from Qdrant (if already stored).
        """
        if force_rebuild:
            # Load files (PDFs, images, etc.) into LlamaIndex Document objects
            docs = load_documents(file_paths)

            # Build and persist index into Qdrant
            self.index = build_index(docs, self.config)
        else:
            try:
                # Attempt to load index from existing Qdrant collection
                self.index = load_existing_index(self.config)
            except Exception as e:
                # If failed, instruct user to rebuild
                raise RuntimeError(
                    "Could not load index from Qdrant. "
                    "If this is your first time running or the collection is empty, "
                    "set `force_rebuild=True`."
                ) from e

        # Set up retriever from index (supports top-k config)
        self.retriever = get_retriever(self.index, self.config.retrieval)

        # Create a default LLM response synthesizer (can be customized)
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.retriever,
            response_synthesizer=ResponseSynthesizer.from_args()
        )

    def query(self, question: str) -> str:
        """
        Ask a question and return the synthesized answer.
        Requires the query engine to be initialized.
        """
        if not self.query_engine:
            raise RuntimeError("Query engine not initialized. Run `load_and_index()` first.")
        return self.query_engine.query(question).response
