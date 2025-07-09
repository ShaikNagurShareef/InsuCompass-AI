import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

from insucompass.config import settings
from insucompass.services.database import setup_database
from insucompass.api.endpoints import router as api_router # Import our API router

# Configure logging for the main application
logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for application startup and shutdown events.
    Ensures database setup runs when the application starts.
    """
    logger.info("Application startup initiated.")
    try:
        setup_database()
        logger.info("Database setup completed during startup.")
    except Exception as e:
        logger.critical(f"Failed to setup database during startup: {e}")
        # Depending on criticality, you might want to raise the exception to prevent startup
        # For now, we log and allow startup, but this might lead to further errors.
    yield
    logger.info("Application shutdown initiated.")

# Initialize the FastAPI application
app = FastAPI(
    title="InsuCompass AI Backend",
    description="Your AI guide to U.S. Health Insurance, powered by Agentic RAG.",
    version="0.1.0",
    lifespan=lifespan # Attach the lifespan context manager
)

# Include our API router
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to InsuCompass AI Backend! Visit /docs for API documentation."}

# Instructions to run the application:
# Save this file as insucompass/main.py
# From your project root directory, run:
# uvicorn insucompass.main:app --reload --host 0.0.0.0 --port 8000
# The --reload flag is useful for development, it reloads the server on code changes.