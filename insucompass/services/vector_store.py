import logging
import chromadb
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from typing import List
from langchain_core.documents import Document

from ..config import settings

logger = logging.getLogger(__name__)

# Use a local, open-source embedding model for cost-effectiveness and privacy.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Define the path for the persistent ChromaDB store
CHROMA_PATH = "data/vector_store"

class VectorStoreService:
    def __init__(self):
        """Initializes the VectorStoreService."""
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.embedding_function = self._get_embedding_function()
        self.collection_name = "insucompass_kb"
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None # LangChain's wrapper handles this
        )
        
        self.langchain_chroma = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )
        logger.info(f"ChromaDB service initialized. Collection '{self.collection_name}' at {CHROMA_PATH}")

    def _get_embedding_function(self) -> Embeddings:
        """Initializes and returns the embedding model."""
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        # Specify 'mps' for Apple Silicon, 'cuda' for NVIDIA, or 'cpu'
        model_kwargs = {'device': 'cpu'} 
        encode_kwargs = {'normalize_embeddings': False}
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Adds a list of documents to the Chroma vector store.

        Args:
            documents: A list of LangChain Document objects.

        Returns:
            A list of vector IDs for the added documents.
        """
        if not documents:
            logger.warning("No documents provided to add to the vector store.")
            return []
            
        logger.info(f"Adding {len(documents)} documents to the vector store...")
        try:
            vector_ids = self.langchain_chroma.add_documents(documents)
            logger.info(f"Successfully added {len(documents)} documents.")
            return vector_ids
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise

    def get_retriever(self, search_kwargs={'k': 5}):
        """Returns a LangChain retriever for the vector store."""
        return self.langchain_chroma.as_retriever(search_type="mmr", search_kwargs=search_kwargs)

# Singleton instance
vector_store_service = VectorStoreService()