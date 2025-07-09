import logging
import json
import uuid
import sqlite3
from typing import List, Dict, Any
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Import all our custom agent and service classes
from insucompass.core.agents.profile_agent import profile_builder
from insucompass.core.agents.query_trasformer import QueryTransformationAgent
from insucompass.core.agents.router_agent import router
from insucompass.services.ingestion_service import IngestionService
from insucompass.core.agents.search_agent import searcher
from insucompass.core.agents.advisor_agent import advisor

from insucompass.services import llm_provider
from insucompass.prompts.prompt_loader import load_prompt
from insucompass.services.vector_store import vector_store_service

llm = llm_provider.get_gemini_llm()
retriever = vector_store_service.get_retriever()
transformer = QueryTransformationAgent(llm, retriever)
ingestor = IngestionService()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Unified LangGraph State Definition ---
class AgentState(TypedDict):
    user_profile: Dict[str, Any]
    user_message: str
    conversation_history: List[str]
    is_profile_complete: bool
    standalone_question: str
    documents: List[Document]
    is_relevant: bool
    generation: str

# --- Graph Nodes ---

def profile_builder_node(state: AgentState) -> Dict[str, Any]:
    """A single turn of the profile building conversation."""
    logger.info("---NODE: PROFILE BUILDER---")
    profile = state["user_profile"]
    message = state["user_message"]
    history = state.get("conversation_history", [])

    if message == "START_PROFILE_BUILDING":
        agent_response = profile_builder.get_next_question(profile, [])
        new_history = [f"Agent: {agent_response}"]
        return {"conversation_history": new_history, "generation": agent_response, "user_profile": profile, "is_profile_complete": False}

    last_question = history[-1][len("Agent: "):] if history and history[-1].startswith("Agent:") else ""
    updated_profile = profile_builder.update_profile_with_answer(profile, last_question, message)
    agent_response = profile_builder.get_next_question(updated_profile, history + [f"User: {message}"])
    
    new_history = history + [f"User: {message}", f"Agent: {agent_response}"]
    
    if agent_response == "PROFILE_COMPLETE":
        logger.info("Profile building complete.")
        final_message = "Great! Your profile is complete. How can I help you with your health insurance questions?"
        new_history[-1] = f"Agent: {final_message}" # Replace "PROFILE_COMPLETE"
        return {"user_profile": updated_profile, "is_profile_complete": True, "conversation_history": new_history, "generation": final_message}
    
    return {"user_profile": updated_profile, "is_profile_complete": False, "conversation_history": new_history, "generation": agent_response}

def reformulate_query_node(state: AgentState) -> Dict[str, Any]:
    """Reformulates the user's question to be self-contained."""
    logger.info("---NODE: REFORMULATE QUERY---")
    question = state["user_message"]
    history = state["conversation_history"]
    user_profile = state["user_profile"]
    
    profile_summary = f"User profile context: State={user_profile.get('state')}, Age={user_profile.get('age')}, History={user_profile.get('medical_history')}"
    prompt = load_prompt("query_reformulator")
    history_str = "\n".join(history)
    
    full_prompt = f"{prompt}\n\n### User Profile Summary\n{profile_summary}\n\n### Conversation History:\n{history_str}\n\n### Follow-up Question:\n{question}"
    
    response = llm.invoke(full_prompt)
    standalone_question = response.content.strip()
    return {"standalone_question": standalone_question}

def retrieve_and_grade_node(state: AgentState) -> Dict[str, Any]:
    """Retrieves documents and grades them."""
    logger.info("---NODE: RETRIEVE & GRADE---")
    standalone_question = state["standalone_question"]
    documents = transformer.transform_and_retrieve(standalone_question)
    is_relevant = router.grade_documents(standalone_question, documents)
    return {"documents": documents, "is_relevant": is_relevant}

def search_and_ingest_node(state: AgentState) -> Dict[str, Any]:
    """Searches the web and ingests new info."""
    logger.info("---NODE: SEARCH & INGEST---")
    web_documents = searcher.search(state["standalone_question"])
    if web_documents:
        ingestor.ingest_documents(web_documents)
    return {}

def generate_answer_node(state: AgentState) -> Dict[str, Any]:
    """Generates the final answer."""
    logger.info("---NODE: GENERATE ADVISOR RESPONSE---")
    generation = advisor.generate_response(
        state["standalone_question"], state["user_profile"], state["documents"]
    )
    history = state["conversation_history"] + [f"User: {state['user_message']}", f"Agent: {generation}"]
    return {"generation": generation, "conversation_history": history}

# --- Conditional Edges ---
def should_search_web(state: AgentState) -> str:
    return "search" if not state["is_relevant"] else "generate"

# (CORRECTED) This is the function for the entry point conditional edge
def decide_entry_point(state: AgentState) -> str:
    """Decides the initial path based on profile completion status."""
    logger.info("---ROUTING: ENTRY POINT---")
    if state.get("is_profile_complete"):
        logger.info(">>> Route: Profile is complete. Starting Q&A.")
        return "qna"
    else:
        logger.info(">>> Route: Profile is not complete. Starting Profile Builder.")
        return "profile"

# --- Build the Graph ---
db_connection = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
memory = SqliteSaver(db_connection)

builder = StateGraph(AgentState)

# (CORRECTED) Removed the faulty entry_router_node
builder.add_node("profile_builder", profile_builder_node)
builder.add_node("reformulate_query", reformulate_query_node)
builder.add_node("retrieve_and_grade", retrieve_and_grade_node)
builder.add_node("search_and_ingest", search_and_ingest_node)
builder.add_node("generate_answer", generate_answer_node)

# (CORRECTED) Set a conditional entry point
builder.set_conditional_entry_point(
    decide_entry_point,
    {
        "profile": "profile_builder",
        "qna": "reformulate_query"
    }
)

# Define graph edges
builder.add_edge("profile_builder", END) # A profile turn is one full loop. The state is saved, and the next call will re-evaluate at the entry point.
builder.add_edge("reformulate_query", "retrieve_and_grade")
builder.add_conditional_edges("retrieve_and_grade", should_search_web, {"search": "search_and_ingest", "generate": "generate_answer"})
builder.add_edge("search_and_ingest", "generate_answer")
builder.add_edge("generate_answer", END)

app = builder.compile(checkpointer=memory)

# --- Interactive Test Harness (CORRECTED) ---
# if __name__ == '__main__':
#     print("--- InsuCompass AI Unified Orchestrator Interactive Test ---")
#     print("Type 'quit' at any time to exit.")

#     test_thread_id = f"interactive-test-{uuid.uuid4()}"
#     thread_config = {"configurable": {"thread_id": test_thread_id}}
#     print(f"Using conversation thread_id: {test_thread_id}")

#     # Initial state for a new user
#     current_state = {
#         "user_profile": {
#             "zip_code": "90210", "county": "Los Angeles", "state": "California", "state_abbreviation": "CA",
#             "age": 45, "gender": "Male", "household_size": 2, "income": 120000,
#             "employment_status": "employed_with_employer_coverage", "citizenship": "US Citizen",
#             "medical_history": None, "medications": None, "special_cases": None
#         },
#         "user_message": "START_PROFILE_BUILDING",
#         "is_profile_complete": False,
#         "conversation_history": [],
#     }

#     while True:
#         print("\n" + "="*20 + " INVOKING GRAPH " + "="*20)
#         print(f"Sending message: '{current_state['user_message']}'")
        
#         # The graph is invoked with the current state
#         final_state = app.invoke(current_state, config=thread_config)

#         # Update our local state from the graph's final output
#         current_state = final_state
#         agent_response = current_state["generation"]
        
#         print(f"\nInsuCompass Agent: {agent_response}")

#         # Get the next input from the user
#         if current_state["is_profile_complete"]:
#             # If the last response was the completion message, prompt for a question
#             if "profile is complete" in agent_response:
#                  next_message = input("Your Question > ")
#             else: # It was a Q&A response, so prompt for another question
#                  next_message = input("Your Follow-up Question > ")
#         else:
#             next_message = input("Your Answer > ")

#         if next_message.lower() == 'quit':
#             print("Exiting test.")
#             break
        
#         # Prepare the state for the next turn
#         current_state["user_message"] = next_message