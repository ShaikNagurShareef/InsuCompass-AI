import re
import logging
import requests
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from tavily import TavilyClient

from insucompass.services import llm_provider
from insucompass.config import settings
from insucompass.prompts.prompt_loader import load_prompt

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

llm = llm_provider.get_gemini_llm()

# Define a dedicated directory for dynamically downloaded files
DYNAMIC_DATA_DIR = Path("data/dynamic")

def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be a valid filename."""
    s = re.sub(r'[<>:"/\\|?*]', '_', name)
    return s.replace(' ', '_')[:150]

def get_file_extension_from_url(url: str) -> str:
    """Intelligently determines the file extension from a URL."""
    # Simple cases first
    if url.lower().endswith('.pdf'):
        return '.pdf'
    if url.lower().endswith('.html') or url.lower().endswith('.htm'):
        return '.html'
    
    # If no clear extension, we assume it's an HTML page
    return '.html'

class SearchAgent:
    """
    An agent that searches the web, saves the results locally in their correct
    format, and returns structured Document objects.
    """

    def __init__(self):
        """Initializes the SearchAgent."""
        try:
            self.query_prompt = load_prompt("search_agent")
            self.tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
            self.session = requests.Session()
            self.session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"})
            DYNAMIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("SearchAgent initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize SearchAgent. Error: {e}")
            raise

    def _formulate_query(self, user_question: str) -> str | None:
        """Uses an LLM to reformulate the user's question into an effective search query."""
        logger.debug(f"Formulating search query for question: '{user_question}'")
        full_prompt = f"{self.query_prompt}\n\nUser Question: {user_question}"
        
        try:
            response = llm.invoke(full_prompt)
            query = response.content.strip()
            
            if query == "NOT_RELEVANT":
                logger.warning("Agent determined question is not relevant for web search.")
                return None
            
            logger.info(f"Formulated search query: '{query}'")
            return query
        except Exception as e:
            logger.error(f"Error during search query formulation: {e}")
            return user_question
        
    def _save_result_to_file(self, result: dict) -> Path | None:
        """Saves a single search result's content to a local file with the correct extension."""
        try:
            url = result.get('url', '')
            # Tavily provides the raw content, which is great for text, but for binary files like PDFs,
            # we must re-download it to get the raw bytes.
            
            if not url:
                return None

            extension = get_file_extension_from_url(url)
            sanitized_url = sanitize_filename(url)
            filename = f"{sanitized_url}{extension}"
            save_path = DYNAMIC_DATA_DIR / filename
            
            # If it's a PDF, we must re-download the raw bytes.
            if extension == '.pdf':
                logger.info(f"Re-downloading PDF content from: {url}")
                response = self.session.get(url, timeout=30, verify=False)
                response.raise_for_status()
                save_path.write_bytes(response.content)
            else:
                # For HTML/text, the content from Tavily is sufficient.
                content = result.get('content', '')
                if not content:
                    return None
                save_path.write_text(content, encoding='utf-8')

            logger.info(f"Saved web content from {url} to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to save search result from {url} to file: {e}")
            return None

    def search(self, user_question: str) -> List[Document]:
        """Executes the full web search process, including saving results to local files."""
        search_query = self._formulate_query(user_question)
        
        if not search_query:
            return []

        logger.info(f"Performing web search with Tavily for query: '{search_query}'")
        try:
            # We ask Tavily to include the raw HTML content in its results
            search_results = self.tavily_client.search(
                query=search_query,
                search_depth="advanced",
                include_raw_content=True, # Important for getting full HTML
                max_results=5
            )
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []

        if not search_results or not search_results.get("results"):
            logger.warning("Tavily search returned no results.")
            return []

        documents = []
        for result in search_results["results"]:
            local_path = self._save_result_to_file(result)
            
            if local_path:
                # The page_content is still useful for the initial grading step,
                # but the ingestion service will use the local_path to load the definitive content.
                doc = Document(
                    page_content=result.get('content', ''),
                    metadata={
                        "source_url": result.get('url', ''),
                        "source_name": result.get('title', 'Web Search Result'),
                        "source_local_path": str(local_path)
                    }
                )
                documents.append(doc)
        
        logger.info(f"Found and saved {len(documents)} documents from the web.")
        return documents
    
searcher = SearchAgent()