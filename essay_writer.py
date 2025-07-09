import os
import sqlite3
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Initialize Groq language model and Tavily client
MODEL = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.environ["GROQ_API_KEY"])
TAVILY_CLIENT = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Initialize SQLite checkpoint database
CONN = sqlite3.connect('checkpoints.db', check_same_thread=False)
CHECKPOINTER = SqliteSaver(CONN)

# Define concise prompts to minimize token usage
PLAN_PROMPT = """You are an expert writer. Create a concise outline for a 5-paragraph essay on the given topic, including brief notes for each section."""

WRITER_PROMPT = """You are an essay assistant. Write a high-quality 5-paragraph essay based on the topic, outline, and provided research content. If a critique is available, revise the draft to address it. Use this information:

{content}"""

REFLECTION_PROMPT = """You are a teacher. Provide concise critique and recommendations for the essay draft, focusing on clarity, depth, structure, and style."""

RESEARCH_PLAN_PROMPT = """You are a researcher. Generate up to 3 concise search queries to gather relevant information for the essay topic."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher. Generate up to 3 concise search queries to address the critique feedback for essay revisions."""

# Define state schema
class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

# Define structured output for search queries
class Queries(BaseModel):
    queries: List[str]

# Token limit for research content (to stay within Groq's ~8k token context window)
MAX_CONTENT_LENGTH = 4000  # Approximate characters, leaving room for prompt and output

def truncate_content(content: List[str], max_length: int = MAX_CONTENT_LENGTH) -> List[str]:
    """Truncates content to fit within token limits."""
    total_length = sum(len(c) for c in content)
    if total_length <= max_length:
        return content
    truncated = []
    current_length = 0
    for item in content:
        if current_length + len(item) <= max_length:
            truncated.append(item)
            current_length += len(item)
        else:
            remaining = max_length - current_length
            truncated.append(item[:remaining])
            break
    return truncated

# Define agent nodes
def plan_node(state: AgentState) -> dict:
    """Creates an essay outline based on the user task."""
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ]
    response = MODEL.invoke(messages)
    return {"plan": response.content}

def research_plan_node(state: AgentState) -> dict:
    """Generates search queries and retrieves content for the essay plan."""
    queries = MODEL.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content = state.get('content', [])
    for query in queries.queries:
        response = TAVILY_CLIENT.search(query=query, max_results=2)
        for result in response['results']:
            content.append(result['content'])
    return {"content": truncate_content(content)}

def generation_node(state: AgentState) -> dict:
    """Generates or revises an essay draft based on the task, plan, and research content."""
    content = "\n\n".join(truncate_content(state.get('content', [])))
    user_message = HumanMessage(content=f"Topic: {state['task']}\nOutline:\n{state['plan']}")
    messages = [
        SystemMessage(content=WRITER_PROMPT.format(content=content)),
        user_message
    ]
    try:
        response = MODEL.invoke(messages)
        return {
            "draft": response.content,
            "revision_number": state.get("revision_number", 1) + 1
        }
    except Exception as e:
        return {
            "draft": f"Error generating draft: {str(e)}",
            "revision_number": state.get("revision_number", 1) + 1
        }

def reflection_node(state: AgentState) -> dict:
    """Critiques the essay draft and provides recommendations for improvement."""
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['draft'])
    ]
    response = MODEL.invoke(messages)
    return {"critique": response.content}

def research_critique_node(state: AgentState) -> dict:
    """Generates search queries and retrieves content to address critique feedback."""
    queries = MODEL.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state.get('content', [])
    for query in queries.queries:
        response = TAVILY_CLIENT.search(query=query, max_results=2)
        for result in response['results']:
            content.append(result['content'])
    return {"content": truncate_content(content)}

def should_continue(state: AgentState) -> str:
    """Determines if the workflow should continue based on revision count."""
    return END if state["revision_number"] > state["max_revisions"] else "reflect"

# Build and compile the LangGraph workflow
def build_graph() -> StateGraph:
    """Constructs and compiles the LangGraph workflow for essay writing."""
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("planner", plan_node)
    builder.add_node("research_plan", research_plan_node)
    builder.add_node("generate", generation_node)
    builder.add_node("reflect", reflection_node)
    builder.add_node("research_critique", research_critique_node)
    
    # Define edges
    builder.set_entry_point("planner")
    builder.add_edge("planner", "research_plan")
    builder.add_edge("research_plan", "generate")
    builder.add_conditional_edges("generate", should_continue, {END: END, "reflect": "reflect"})
    builder.add_edge("reflect", "research_critique")
    builder.add_edge("research_critique", "generate")
    
    return builder.compile(checkpointer=CHECKPOINTER)

def save_graph_as_png(graph: StateGraph, output_path: str = "workflow_graph.png") -> None:
    """Saves the LangGraph workflow as a PNG image."""
    try:
        png_data = graph.get_graph().draw_png()
        with open(output_path, "wb") as f:
            f.write(png_data)
        print(f"Graph saved as {output_path}")
    except Exception as e:
        print(f"Error saving graph: {e}")

# Main execution
def main():
    """Runs the essay writing workflow and saves the graph visualization."""
    graph = build_graph()
    
    # Save the graph visualization
    save_graph_as_png(graph, "essay_workflow.png")
    
    # Run the workflow
    thread = {"configurable": {"thread_id": "1"}}
    task = {
        "task": "AI in Medical",
        "max_revisions": 2,
        "revision_number": 1
    }
    
    for state in graph.stream(task, thread):
        print(state)

if __name__ == "__main__":
    main()