import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import List

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """
    Loads and validates application settings from environment variables.
    """
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # JWT Settings
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "default_secret")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

    # Database
    DATABASE_URL: str = "insucompass.db"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # LLM Settings

    GROQ_MODEL_NAME: str = "llama-3.3-70b-versatile"
    GROQ_FAST_MODEL_NAME: str = "llama-3.1-8b-instant"
    GEMINI_PRO_MODEL_NAME: str = "gemini-2.5-pro"
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash" #"gemini-2.5-flash" # "gemini-2.0-flash"
    GEMINI_FAST_MODEL_NAME: str = "gemini-2.5-flash-lite-preview-06-17"

    # CRAWLING JOBS CONFIGURATION
    CRAWLING_JOBS: List[dict] = [
        {
            "name": "HealthCare.gov Crawl",
            "start_url": "https://www.healthcare.gov/",
            "method": "requests_crawl",
            "domain_lock": "www.healthcare.gov",
            "crawl_depth": 2,
            "content_types": ["pdf", "html"],
            "status": "active"
        },
        {
            "name": "CMS.gov Regulations & Guidance Crawl",
            "start_url": "https://www.cms.gov/regulations-and-guidance",
            "method": "selenium_crawl",
            "domain_lock": "www.cms.gov",
            "crawl_depth": 2,
            "content_types": ["pdf", "html"],
            "status": "active"
        },
        {
            "name": "Medicaid.gov Crawl",
            "start_url": "https://www.medicaid.gov/",
            "method": "requests_crawl",
            "domain_lock": "www.medicaid.gov",
            "crawl_depth": 2,
            "content_types": ["pdf", "html"],
            "status": "active"
        },
        {
            "name": "Medicare.gov Crawl",
            "start_url": "https://www.medicare.gov/",
            "method": "selenium_crawl",
            "domain_lock": "www.medicare.gov",
            "crawl_depth": 1,
            "content_types": ["html"],
            "status": "active"
        },
        {
            "name": "TRICARE Publications Crawl",
            "start_url": "https://www.tricare.mil/publications",
            "method": "selenium_crawl",
            "domain_lock": "www.tricare.mil",
            "crawl_depth": 1,
            "content_types": ["pdf", "html"],
            "status": "active"
        },
        {
            "name": "VA.gov Health Benefits Crawl",
            "start_url": "https://www.va.gov/health-care/",
            "method": "requests_crawl",
            "domain_lock": "www.va.gov",
            "crawl_depth": 2,
            "content_types": ["html"],
            "status": "active"
        },
    ]

    class Config:
        case_sensitive = True

# Instantiate settings
settings = Settings()