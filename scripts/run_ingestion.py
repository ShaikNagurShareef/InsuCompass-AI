import logging
import json
from insucompass.services.database import get_db_connection
from insucompass.services.vector_store import vector_store_service
from scripts.data_processing.document_loader import load_document
from scripts.data_processing.chunker import chunk_text
from insucompass.config import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_source_for_ingestion(source: dict):
    """
    Loads, chunks, and embeds a single data source.
    """
    source_id = source['id']
    local_path = source['local_path']
    
    logger.info(f"Starting ingestion for source_id: {source_id}, path: {local_path}")
    
    # 1. Load document content
    text_content = load_document(local_path)
    if not text_content:
        logger.error(f"Could not load content from {local_path}. Skipping ingestion for this source.")
        # Update status in DB to 'ingestion_failed'
        with get_db_connection() as conn:
            conn.cursor().execute("UPDATE data_sources SET status = ? WHERE id = ?", ('ingestion_failed', source_id))
            conn.commit()
        return

    # 2. Chunk the text with metadata
    documents = chunk_text(text_content, source_metadata=source)
    if not documents:
        logger.warning(f"No chunks were created for source_id: {source_id}. Skipping embedding.")
        return

    # 3. Add documents to the vector store
    try:
        vector_ids = vector_store_service.add_documents(documents)
    except Exception as e:
        logger.error(f"Failed to embed documents for source_id {source_id}: {e}")
        with get_db_connection() as conn:
            conn.cursor().execute("UPDATE data_sources SET status = ? WHERE id = ?", ('embedding_failed', source_id))
            conn.commit()
        return

    if len(vector_ids) != len(documents):
        logger.error(f"Mismatch between number of documents ({len(documents)}) and returned vector IDs ({len(vector_ids)}). Aborting DB update for this source.")
        return

    # 4. Store chunk info and vector IDs in SQLite
    logger.info(f"Storing {len(documents)} chunk records in the database...")
    insert_query = "INSERT INTO knowledge_chunks (source_id, chunk_text, metadata_json, vector_id) VALUES (?, ?, ?, ?)"
    chunk_data_to_insert = [
        (
            source_id,
            doc.page_content,
            json.dumps(doc.metadata),
            vec_id
        ) for doc, vec_id in zip(documents, vector_ids)
    ]
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany(insert_query, chunk_data_to_insert)
        # Update the source status to 'ingested'
        cursor.execute("UPDATE data_sources SET status = ? WHERE id = ?", ('ingested', source_id))
        conn.commit()
    logger.info(f"Successfully ingested source_id: {source_id}")


def main():
    """
    Main function to run the ingestion pipeline.
    It finds all downloaded documents that haven't been ingested yet
    and processes them.
    """
    logger.info("--- Starting InsuCompass AI Data Ingestion Pipeline ---")
    
    # Find sources that have been downloaded but not yet ingested
    # Statuses 'processed' and 'updated' are from the crawler step.
    query = "SELECT * FROM data_sources WHERE status IN ('processed', 'updated') AND local_path IS NOT NULL"
    
    with get_db_connection() as conn:
        sources_to_ingest = conn.cursor().execute(query).fetchall()

    if not sources_to_ingest:
        logger.info("No new or updated sources to ingest. Pipeline finished.")
        return

    logger.info(f"Found {len(sources_to_ingest)} sources to ingest.")
    
    for source_row in sources_to_ingest:
        try:
            process_source_for_ingestion(dict(source_row))
        except Exception as e:
            logger.error(f"A critical error occurred while processing source_id {source_row['id']}: {e}", exc_info=True)
            with get_db_connection() as conn:
                conn.cursor().execute("UPDATE data_sources SET status = ? WHERE id = ?", ('ingestion_failed', source_row['id']))
                conn.commit()

    logger.info("--- Data Ingestion Pipeline Finished ---")

if __name__ == "__main__":
    main()