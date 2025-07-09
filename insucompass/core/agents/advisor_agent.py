import logging
import json
from typing import List, Dict, Any
from langchain_core.documents import Document

from insucompass.services import llm_provider
from insucompass.config import settings
from insucompass.prompts.prompt_loader import load_prompt

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

llm = llm_provider.get_gemini_pro_llm()

class AdvisorAgent:
    """
    The final agent in the Q&A pipeline. It uses a Chain-of-Thought process
    to synthesize the user's profile and retrieved context into a comprehensive,
    personalized, and conversational response that includes a follow-up question.
    """

    def __init__(self):
        """Initializes the AdvisorAgent."""
        try:
            self.agent_prompt = load_prompt("advisor_agent")
            logger.info("Conversational AdvisorAgent initialized successfully.")
        except FileNotFoundError:
            logger.critical("AdvisorAgent prompt file not found. The agent cannot function.")
            raise

    def generate_response(
        self,
        question: str,
        user_profile: Dict[str, Any],
        documents: List[Document]
    ) -> str:
        """
        Generates the final, synthesized, and conversational response.

        Args:
            question: The user's original question.
            user_profile: The user's complete profile.
            documents: The relevant documents retrieved from the knowledge base.

        Returns:
            A string containing the final, formatted answer and a follow-up question.
        """
        if not documents:
            logger.warning("AdvisorAgent received no documents. Cannot generate a grounded response.")
            return "I'm sorry, but I couldn't find any relevant information to answer your question, even after searching the web. Is there another way I can help you look into this?"

        # Prepare the context for the prompt
        profile_str = json.dumps(user_profile, indent=2)
        context_str = "\n\n---\n\n".join(
            [f"[METADATA: source_name='{d.metadata.get('source_name', 'N/A')}', source_url='{d.metadata.get('source_url', 'N/A')}']\n\n{d.page_content}" for d in documents]
        )
        
        full_prompt = (
            f"{self.agent_prompt}\n\n"
            f"### CONTEXT FOR YOUR RESPONSE\n"
            f"user_profile: {profile_str}\n\n"
            f"user_question: \"{question}\"\n\n"
            f"retrieved_context:\n{context_str}"
        )
        
        logger.info("Generating final conversational response with AdvisorAgent...")
        try:
            response = llm.invoke(full_prompt)
            generation = response.content.strip()
            logger.info("Successfully generated final conversational answer.")
            return generation
        except Exception as e:
            logger.error(f"Error during final answer generation: {e}")
            return "I apologize, I encountered an error while trying to formulate the final answer. Please try again."
        
advisor = AdvisorAgent()