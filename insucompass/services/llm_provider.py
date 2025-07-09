import logging
from langchain_groq import ChatGroq 
from langchain_google_genai import ChatGoogleGenerativeAI

from insucompass.config import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GROQ_MODEL_NAME = settings.GROQ_MODEL_NAME
GROQ_FAST_MODEL_NAME = settings.GROQ_FAST_MODEL_NAME
GEMINI_PRO_MODEL_NAME = settings.GEMINI_PRO_MODEL_NAME
GEMINI_MODEL_NAME = settings.GEMINI_MODEL_NAME
GEMINI_FAST_MODEL_NAME = settings.GEMINI_FAST_MODEL_NAME

def get_gemini_pro_llm():
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_PRO_MODEL_NAME,
        temperature=0.1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    logger.info(f"Initialized LLM Provider: {GEMINI_MODEL_NAME}")
    return llm

def get_gemini_llm():
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        temperature=0.1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    logger.info(f"Initialized LLM Provider: {GEMINI_MODEL_NAME}")
    return llm

def get_gemini_fast_llm():
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_FAST_MODEL_NAME,
        temperature=0.1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    logger.info(f"Initialized LLM Provider: {GEMINI_FAST_MODEL_NAME}")
    return llm


def get_llama_llm():
    llm = ChatGroq(
        temperature=0.1, # Lower temperature for factual, consistent outputs
        groq_api_key=settings.GROQ_API_KEY,
        model_name=GROQ_MODEL_NAME
    )
    logger.info(f"Initialized LLM Provider: {GROQ_MODEL_NAME}")
    return llm

def get_llama_fast_llm():
    llm = ChatGroq(
        temperature=0.1, # Lower temperature for factual, consistent outputs
        groq_api_key=settings.GROQ_API_KEY,
        model_name=GROQ_FAST_MODEL_NAME
    )
    logger.info(f"Initialized Fast LLM Provider: {GROQ_FAST_MODEL_NAME}")
    return llm