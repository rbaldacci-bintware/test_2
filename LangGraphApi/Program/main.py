# main.py
from fastapi import FastAPI
from api import table_extraction_router, semantic_chunking_router

app = FastAPI(
    title="LangGraph Agent API",
    description="API per estrazione tabelle e chunking semantico con LangGraph.",
    version="2.0.0"
)

# Router per estrazione tabelle (primo agente)
app.include_router(
    table_extraction_router.router, 
    prefix="/api/v1", 
    tags=["1. Table Extraction"]
)

# Router per chunking semantico (secondo agente)
app.include_router(
    semantic_chunking_router.router, 
    prefix="/api/v1", 
    tags=["2. Semantic Chunking"]
)

@app.get("/")
def read_root():
    return {
        "message": "Benvenuto nella LangGraph Agent API",
        "version": "2.0.0",
        "endpoints": {
            "table_extraction": "/api/v1/extract-table",
            "semantic_chunking": "/api/v1/semantic-chunking",
            "docs": "/docs"
        }
    }

# Per avviare il server, esegui da terminale:
# uvicorn main:app --reload