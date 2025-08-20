# /Program/agents/pdf_chunking_agent.py
import io
import uuid
import logging
from typing import List, Dict, Any, Literal, TypedDict

# LangGraph e LangChain
from langgraph.graph import StateGraph, END
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Utility per PDF
import fitz  # PyMuPDF
import pdfplumber

# IMPORTANTE: Importa il tuo agente di chunking semantico esistente
from agents.semantic_chunking_agent import chunking_agent

# Configurazione del logger
logger = logging.getLogger(__name__)

# --- Stato del Grafo Orchestratore ---
class PDFChunkingState(TypedDict):
    pdf_file: bytes
    pdf_name: str
    pdf_metadata: Dict[str, Any]
    raw_text_pages: List[Dict]
    detected_tables: List[Dict]
    text_chunks: List[Dict]
    table_chunks: List[Dict]
    final_chunks: List[Dict]
    processing_strategy: str
    error_message: str

# --- Nodi del Grafo Orchestratore ---

def pdf_analyzer_node(state: PDFChunkingState) -> Dict:
    """Analizza il PDF per determinare la strategia di elaborazione."""
    logger.info(">>> NODO (Orchestratore): Analisi PDF...")
    try:
        doc = fitz.open(stream=state["pdf_file"], filetype="pdf")
        has_text = any(page.get_text().strip() for page in doc)
        
        has_tables = False
        with pdfplumber.open(io.BytesIO(state["pdf_file"])) as pdf:
            has_tables = any(page.extract_tables() for page in pdf.pages)

        if has_text and has_tables: strategy = "mixed_content"
        elif has_text: strategy = "text_only"
        elif has_tables: strategy = "tables_only"
        else: strategy = "no_content"
        
        metadata = {"pages": len(doc), "has_tables": has_tables, "has_text": has_text}
        logger.info(f"    Strategia determinata: {strategy}")
        return {"pdf_metadata": metadata, "processing_strategy": strategy}
    except Exception as e:
        return {"error_message": f"Errore in pdf_analyzer_node: {e}"}

def text_extractor_node(state: PDFChunkingState) -> Dict:
    """Estrae il testo grezzo dal PDF."""
    logger.info(">>> NODO (Orchestratore): Estrazione Testo...")
    doc = fitz.open(stream=state["pdf_file"], filetype="pdf")
    raw_pages = [{"page_num": i + 1, "content": page.get_text("text")} for i, page in enumerate(doc)]
    return {"raw_text_pages": raw_pages}

def table_extractor_node(state: PDFChunkingState) -> Dict:
    """Estrae le tabelle grezze dal PDF per l'agente semantico."""
    logger.info(">>> NODO (Orchestratore): Estrazione Tabelle...")
    detected_tables = []
    with pdfplumber.open(io.BytesIO(state["pdf_file"])) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table_data in tables:
                if not table_data: continue
                cells = []
                for row_idx, row in enumerate(table_data):
                    for col_idx, cell_value in enumerate(row):
                        cells.append({"rowIndex": row_idx + 1, "columnIndex": col_idx + 1, "value": str(cell_value or "")})
                
                detected_tables.append({
                    "page_num": i + 1,
                    "rows": len(table_data),
                    "columns": len(table_data[0]),
                    "extracted_cells": cells
                })
    return {"detected_tables": detected_tables}

def text_chunker_node(state: PDFChunkingState) -> Dict:
    """Crea chunk dal testo."""
    logger.info(">>> NODO (Orchestratore): Chunking Testo...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
    text_chunks = []
    for page in state.get("raw_text_pages", []):
        if page["content"].strip():
            chunks = splitter.split_text(page["content"])
            for chunk_text in chunks:
                text_chunks.append({"content": chunk_text, "metadata": {"source_page": page["page_num"], "chunk_type": "text"}})
    return {"text_chunks": text_chunks}

def table_chunker_node(state: PDFChunkingState) -> Dict:
    """Invoca il tuo semantic_chunking_agent per ogni tabella."""
    logger.info(">>> NODO (Orchestratore): Chunking Tabelle (invoca il tuo agente)...")
    all_table_chunks = []
    for table in state.get("detected_tables", []):
        agent_input = {
            "file_name": state["pdf_name"],
            "rows": table["rows"],
            "columns": table["columns"],
            "extracted_cells": table["extracted_cells"],
            "regeneration_count": 0,
        }
        config = {"configurable": {"thread_id": f"table_chunker_{uuid.uuid4()}"}}
        
        # *** CHIAMATA AL TUO AGENTE ESISTENTE ***
        final_agent_state = chunking_agent.invoke(agent_input, config=config)
        
        agent_chunks = final_agent_state.get("chunks", [])
        for chunk in agent_chunks:
            chunk["metadata"]["source_page"] = table["page_num"]
        all_table_chunks.extend(agent_chunks)
        
    return {"table_chunks": all_table_chunks}

def chunk_merger_node(state: PDFChunkingState) -> Dict:
    """Unifica e ordina tutti i chunk."""
    logger.info(">>> NODO (Orchestratore): Unione Chunks...")
    all_chunks = state.get("text_chunks", []) + state.get("table_chunks", [])
    all_chunks.sort(key=lambda x: x["metadata"].get("source_page", 0))
    for i, chunk in enumerate(all_chunks):
        chunk["metadata"]["global_index"] = i
    return {"final_chunks": all_chunks}

# --- Grafo Orchestratore ---
def create_orchestrator_agent():
    workflow = StateGraph(PDFChunkingState)
    workflow.add_node("pdf_analyzer", pdf_analyzer_node)
    workflow.add_node("text_extractor", text_extractor_node)
    workflow.add_node("table_extractor", table_extractor_node)
    workflow.add_node("text_chunker", text_chunker_node)
    workflow.add_node("table_chunker", table_chunker_node)
    workflow.add_node("chunk_merger", chunk_merger_node)
    
    workflow.set_entry_point("pdf_analyzer")

    def decide_strategy(state: PDFChunkingState):
        if state.get("error_message"): return "end"
        return state["processing_strategy"]

    workflow.add_conditional_edges(
        "pdf_analyzer",
        decide_strategy,
        {
            "mixed_content": "text_extractor",
            "text_only": "text_extractor",
            "tables_only": "table_extractor",
            "no_content": END,
            "end": END,
        }
    )
    
    # Flusso per contenuti misti
    workflow.add_edge("text_extractor", "table_extractor")
    workflow.add_edge("table_extractor", "text_chunker")
    workflow.add_edge("text_chunker", "table_chunker")
    workflow.add_edge("table_chunker", "chunk_merger")
    
    # Flusso per solo testo (salta i nodi delle tabelle)
    def route_after_text_extraction(state: PDFChunkingState):
      return "tables_only" if not state.get("pdf_metadata", {}).get("has_text") else "mixed_content"

    workflow.add_conditional_edges(
        "text_extractor",
        lambda s: "text_chunker" if s["processing_strategy"] == "text_only" else "table_extractor"
    )
    workflow.add_edge("text_chunker", "chunk_merger")

    # Flusso per solo tabelle (salta i nodi del testo)
    workflow.add_edge("table_extractor", "table_chunker")
    workflow.add_edge("table_chunker", "chunk_merger")
    
    workflow.add_edge("chunk_merger", END)
    
    return workflow.compile()

# Esponi una singola istanza compilata dell'agente orchestratore
orchestrator_agent = create_orchestrator_agent()