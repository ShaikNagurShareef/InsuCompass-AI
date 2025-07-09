import logging
from typing import List
from pathlib import Path
from langchain_core.documents import Document
import hashlib

from insucompass.config import settings
from insucompass.services.database import find_or_create_web_source
from insucompass.services.vector_store import vector_store_service

from scripts.data_processing.chunker import chunk_text
from scripts.data_processing.document_loader import load_document

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IngestionService:
    """
    Handles the dynamic ingestion of new documents from local file paths.
    """

    def ingest_documents(self, documents_from_search: List[Document]):
        """
        Processes and ingests a list of documents found by the search agent.
        It reads the content from the local path specified in the metadata.
        """
        if not documents_from_search:
            logger.info("No documents provided for ingestion.")
            return

        logger.info(f"Starting dynamic ingestion of {len(documents_from_search)} documents...")
        
        all_chunks_to_embed = []

        for doc_meta in documents_from_search:
            source_url = doc_meta.metadata.get("source_url")
            source_name = doc_meta.metadata.get("source_name", "Unknown Web Source")
            local_path_str = doc_meta.metadata.get("source_local_path")

            if not all([source_url, local_path_str]):
                logger.warning(f"Skipping document due to missing metadata: {doc_meta.metadata}")
                continue

            # 1. Register the source in SQLite to get a source_id
            try:
                source_id = find_or_create_web_source(url=source_url, name=source_name)
            except Exception as e:
                logger.error(f"Failed to register source {source_url} in database: {e}")
                continue

            # 2. Load the full document content from the saved local file
            # We use our existing document_loader for this.
            full_doc = load_document(local_path_str)
            if not full_doc:
                logger.warning(f"Could not load content from local path {local_path_str}. Skipping.")
                continue

            # 3. Chunk the full document
            chunks = chunk_text(full_doc, doc_meta.metadata)
            
            # 4. Enrich metadata for each chunk
            for i, chunk in enumerate(chunks):
                hash_id = hashlib.md5(f"{source_url}_{i}".encode()).hexdigest()
                chunk_id = f"dynamic_{hash_id}"
                
                chunk.metadata['source_id'] = source_id
                chunk.metadata['chunk_id'] = chunk_id
                chunk.metadata['source_url'] = source_url
                chunk.metadata['source_name'] = source_name
                chunk.metadata['source_local_path'] = local_path_str
                
                all_chunks_to_embed.append(chunk)

        # 5. Embed and store in ChromaDB
        if all_chunks_to_embed:
            logger.info(f"Embedding and storing {len(all_chunks_to_embed)} new chunks in ChromaDB.")
            try:
                vector_store_service.add_documents(all_chunks_to_embed)
                logger.info("Dynamic ingestion completed successfully.")
            except Exception as e:
                logger.error(f"Failed to add chunks to vector store during dynamic ingestion: {e}")
        else:
            logger.info("No chunks generated during dynamic ingestion.")