# main.py
from fastapi import FastAPI
from api import table_extraction_router, semantic_chunking_router, pdf_chunking_router, hierarchical_chunking_router

app = FastAPI(
    title="LangGraph Agent API",
    description="API per estrazione tabelle, chunking semantico e chunking gerarchico con LangGraph.",
    version="3.0.0"
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

# Router per orchestrator chunking (terzo agente)
app.include_router(
    pdf_chunking_router.router,
    prefix="/api/v1", 
    tags=["3. Orchestrator Chunking"]
)

# Router per hierarchical chunking (quarto agente - NUOVO)
app.include_router(
    hierarchical_chunking_router.router,
    prefix="/api/v1", 
    tags=["4. Hierarchical Chunking"]
)

@app.get("/")
def read_root():
    return {
        "message": "Benvenuto nella LangGraph Agent API",
        "version": "3.0.0",
        "endpoints": {
            "hierarchical_chunking": "/api/v1/hierarchical-chunking",  # NUOVO
            "pdf_chunking": "/api/v1/pdf-chunking",
            "table_extraction": "/api/v1/extract-table",
            "semantic_chunking": "/api/v1/semantic-chunking",
            "docs": "/docs"
        },
        "description": {
            "table_extraction": "Estrae tabelle da PDF",
            "semantic_chunking": "Crea chunk semantici da tabelle",
            "pdf_chunking": "Orchestrazione completa PDF->chunks",
            "hierarchical_chunking": "Chunking avanzato con struttura gerarchica completa"
        }
    }

# Per avviare il server, esegui da terminale:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000