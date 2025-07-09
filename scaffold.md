insucompass-ai/
├── .env
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── poetry.lock
├── README.md
│
├── insucompass/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration loader
│   │
│   ├── api/                    # FastAPI routers and dependencies
│   │   ├── __init__.py
│   │   ├── endpoints.py
│   │   └── security.py
│   │
│   ├── prompts/                # Prompts folder
│   │   └── prompts.py          # Prompts file
│   ├── core/                   # Core business logic
│   │   ├── __init__.py
│   │   ├── agent_orchestrator.py # LangGraph definition and state
│   │   └── agents.py             # Individual agent node implementations
│   │
│   └── services/               # External service clients
│       ├── __init__.py
│       ├── vector_store.py       # ChromaDB client
│       ├── database.py           # SQLite connection and models
│       └── llm_provider.py       # Groq client
│
├── scripts/
│   ├── __init__.py
│   ├── run_crawler.py          # Script to execute the data crawling
│   ├── run_ingestion.py        # Script to run the full ingestion pipeline
│   │
│   └── data_processing/
│       ├── __init__.py
│       ├── crawler_utils.py      # Crawling utils
│       ├── document_loader       # Documents loader utility
│       ├── crawler.py            # Web crawling logic
│       ├── chunker.py            # Text splitting and chunking logic
│       └── embedder.py           # Embedding generation logic
│
├── frontend/
│   └── app.py                  # Streamlit application
│
└── tests/
    ├── __init__.py
    ├── test_api.py
    └── test_agents.py