import logging
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any

# Import our services, agents, and models
from insucompass.services.zip_client import get_geo_data_from_zip
from insucompass.core.models import GeoDataResponse, ChatRequest, ChatResponse
from insucompass.core.agents.profile_agent import profile_builder
from insucompass.core.agent_orchestrator import app as orchestrator # The compiled LangGraph app

from insucompass.services.database import get_db_connection, create_or_update_user_profile, get_user_profile

# Configure logging
logger = logging.getLogger(__name__)

# Create the API router
router = APIRouter()
logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/geodata/{zip_code}", response_model=GeoDataResponse)
def get_geolocation_data(zip_code: str):
    """Endpoint to get county, city, and state information from a given ZIP code."""
    if not zip_code.isdigit() or len(zip_code) != 5:
        raise HTTPException(status_code=400, detail="Invalid ZIP code format.")
    geo_data = get_geo_data_from_zip(zip_code)
    if not geo_data:
        raise HTTPException(status_code=404, detail="Could not find location data.")
    return GeoDataResponse(
        zip_code=zip_code, county=geo_data.county.replace(" County", ""),
        city=geo_data.city, state=geo_data.state, state_abbreviation=geo_data.state_abbr
    )

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handles the entire conversational flow using the unified LangGraph orchestrator.
    The orchestrator manages the state, including profile building and Q&A.
    """
    logger.info(f"Received unified chat request for thread_id: {request.thread_id}")
    
    # LangGraph needs a thread_id to save/load conversation state from its checkpointer
    thread_config = {"configurable": {"thread_id": request.thread_id}}
    
    # The graph's state is loaded automatically by LangGraph using the thread_id.
    # We only need to provide the inputs for the current turn.
    inputs = {
        "user_profile": request.user_profile,
        "user_message": request.message,
        "is_profile_complete": request.is_profile_complete,
        "conversation_history": request.conversation_history,
    }

    try:
        # We invoke the graph. It will load the previous state, run the necessary nodes,
        # and save the new state, all in one call.
        final_state = orchestrator.invoke(inputs, config=thread_config)
        
        # Extract the relevant data from the final state of the graph
        agent_response = final_state.get("generation")
        updated_profile = final_state.get("user_profile")
        updated_history = final_state.get("conversation_history")
        is_profile_complete = final_state.get("is_profile_complete")
        
        if not agent_response:
            agent_response = "I'm sorry, I encountered an issue. Could you please rephrase?"

        logger.info("Unified graph execution completed successfully.")
        
        return ChatResponse(
            agent_response=agent_response,
            updated_profile=updated_profile,
            updated_history=updated_history,
            is_profile_complete=is_profile_complete
        )

    except Exception as e:
        logger.error(f"Error during unified graph orchestration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred in the AI orchestrator.")