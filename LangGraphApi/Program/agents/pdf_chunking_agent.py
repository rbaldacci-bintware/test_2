# /Program/agents/pdf_chunking_agent.py
import io
import uuid
import logging
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Literal, TypedDict
import re

# LangGraph
from langgraph.graph import StateGraph, END

# Utility per PDF
import fitz  # PyMuPDF
import pdfplumber

# Google AI per chunking intelligente
import google.generativeai as genai
import os

# Importa il tuo agente di chunking semantico esistente
from agents.semantic_chunking_agent import chunking_agent

# Configurazione del logger
logger = logging.getLogger(__name__)

# Configura Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Stato del Grafo Orchestratore ---
class PDFChunkingState(TypedDict):
    pdf_file: bytes
    pdf_name: str
    pdf_metadata: Dict[str, Any]
    page_classifications: List[Dict]
    raw_text_blocks: List[Dict]  # Blocchi di testo estratti
    detected_tables: List[Dict]
    text_chunks: List[Dict]
    table_chunks: List[Dict]
    final_chunks: List[Dict]
    processing_strategy: str
    error_message: str

# --- Nodi del Grafo Orchestratore ---

def pdf_analyzer_node(state: PDFChunkingState) -> Dict:
    """Analizza il PDF per determinare la strategia di elaborazione."""
    logger.info(">>> NODO: Analisi PDF...")
    try:
        doc = fitz.open(stream=state["pdf_file"], filetype="pdf")
        
        page_classifications = []
        has_text = False
        has_tables = False
        total_text_length = 0
        
        with pdfplumber.open(io.BytesIO(state["pdf_file"])) as pdf:
            for i, (fitz_page, plumber_page) in enumerate(zip(doc, pdf.pages)):
                # Estrai testo completo dalla pagina
                page_text = fitz_page.get_text().strip()
                
                # Estrai tabelle
                page_tables = plumber_page.extract_tables()
                
                # Calcola quanto testo c'è FUORI dalle tabelle
                non_table_text = page_text
                if page_tables:
                    for table in page_tables:
                        if table:
                            for row in table:
                                if row:
                                    row_text = " ".join(str(cell) for cell in row if cell)
                                    non_table_text = non_table_text.replace(row_text, "")
                
                non_table_text = non_table_text.strip()
                
                page_has_text = len(non_table_text) > 50  # Almeno 50 caratteri di testo non tabellare
                page_has_tables = bool(page_tables)
                
                has_text = has_text or page_has_text
                has_tables = has_tables or page_has_tables
                total_text_length += len(non_table_text)
                
                page_classifications.append({
                    "page_num": i + 1,
                    "has_text": page_has_text,
                    "has_tables": page_has_tables,
                    "table_count": len(page_tables) if page_tables else 0,
                    "text_length": len(non_table_text),
                    "dominant_type": "mixed" if page_has_text and page_has_tables else 
                                    "text" if page_has_text else 
                                    "tables" if page_has_tables else "empty"
                })

        # Determina strategia globale
        if has_text and has_tables:
            strategy = "mixed_content"
        elif has_text:
            strategy = "text_only"
        elif has_tables:
            strategy = "tables_only"
        else:
            strategy = "no_content"
        
        metadata = {
            "pages": len(doc),
            "has_text": has_text,
            "has_tables": has_tables,
            "total_text_length": total_text_length,
            "page_count_with_text": sum(1 for p in page_classifications if p["has_text"]),
            "page_count_with_tables": sum(1 for p in page_classifications if p["has_tables"])
        }
        
        logger.info(f"    Strategia: {strategy}")
        logger.info(f"    Testo totale (non tabellare): {total_text_length} caratteri")
        
        return {
            "pdf_metadata": metadata,
            "page_classifications": page_classifications,
            "processing_strategy": strategy
        }
    except Exception as e:
        logger.error(f"Errore in pdf_analyzer: {e}")
        return {"error_message": f"Errore in pdf_analyzer_node: {e}"}

def text_extractor_node(state: PDFChunkingState) -> Dict:
    """Estrae blocchi di testo significativi escludendo le tabelle."""
    logger.info(">>> NODO: Estrazione Testo (escludendo tabelle)...")
    
    doc = fitz.open(stream=state["pdf_file"], filetype="pdf")
    text_blocks = []
    
    with pdfplumber.open(io.BytesIO(state["pdf_file"])) as pdf:
        accumulated_text = ""
        current_page_start = 1
        
        for i, (fitz_page, plumber_page) in enumerate(zip(doc, pdf.pages)):
            page_class = state["page_classifications"][i]
            
            # Estrai tutto il testo
            full_text = fitz_page.get_text("text")
            
            # Rimuovi contenuto delle tabelle se presenti
            if page_class["has_tables"]:
                tables = plumber_page.extract_tables()
                for table in tables:
                    if table:
                        for row in table:
                            if row:
                                # Rimuovi ogni cella dal testo
                                for cell in row:
                                    if cell:
                                        full_text = full_text.replace(str(cell), " ")
            
            # Pulisci il testo
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            # Se c'è testo significativo, accumulalo
            if len(full_text) > 50:
                if accumulated_text:
                    accumulated_text += "\n\n"
                accumulated_text += full_text
                
                # Se l'accumulo supera i 2000 caratteri o siamo all'ultima pagina, crea un blocco
                if len(accumulated_text) > 2000 or i == len(pdf.pages) - 1:
                    if accumulated_text.strip():
                        text_blocks.append({
                            "content": accumulated_text.strip(),
                            "start_page": current_page_start,
                            "end_page": i + 1,
                            "length": len(accumulated_text)
                        })
                    accumulated_text = ""
                    current_page_start = i + 2
        
        # Aggiungi eventuale testo rimanente
        if accumulated_text.strip():
            text_blocks.append({
                "content": accumulated_text.strip(),
                "start_page": current_page_start,
                "end_page": len(pdf.pages),
                "length": len(accumulated_text)
            })
    
    logger.info(f"    Estratti {len(text_blocks)} blocchi di testo")
    for block in text_blocks:
        logger.info(f"      Blocco pagine {block['start_page']}-{block['end_page']}: {block['length']} caratteri")
    
    return {"raw_text_blocks": text_blocks}

def table_extractor_node(state: PDFChunkingState) -> Dict:
    """Estrae le tabelle dal PDF."""
    logger.info(">>> NODO: Estrazione Tabelle...")
    detected_tables = []
    
    with pdfplumber.open(io.BytesIO(state["pdf_file"])) as pdf:
        for i, page in enumerate(pdf.pages):
            page_class = state["page_classifications"][i]
            
            if not page_class["has_tables"]:
                continue
            
            tables = page.extract_tables()
            for j, table_data in enumerate(tables):
                if not table_data or len(table_data) == 0:
                    continue
                
                cells = []
                for row_idx, row in enumerate(table_data):
                    for col_idx, cell_value in enumerate(row):
                        cells.append({
                            "rowIndex": row_idx + 1,
                            "columnIndex": col_idx + 1,
                            "value": str(cell_value or "")
                        })
                
                detected_tables.append({
                    "page_num": i + 1,
                    "table_index": j,
                    "rows": len(table_data),
                    "columns": len(table_data[0]) if table_data else 0,
                    "extracted_cells": cells
                })
    
    logger.info(f"    Estratte {len(detected_tables)} tabelle")
    return {"detected_tables": detected_tables}

def ai_text_chunker_node(state: PDFChunkingState) -> Dict:
    """Usa AI per creare chunk semantici intelligenti dal testo."""
    logger.info(">>> NODO: Chunking Testo con AI...")
    
    text_blocks = state.get("raw_text_blocks", [])
    if not text_blocks:
        logger.info("    Nessun blocco di testo da processare")
        return {"text_chunks": []}
    
    all_text_chunks = []
    
    for block_idx, text_block in enumerate(text_blocks):
        content = text_block["content"]
        
        # Se il blocco è molto corto, crealo come chunk singolo
        if len(content) < 500:
            chunk = {
                "content": content,
                "metadata": {
                    "chunk_id": f"{state['pdf_name']}_text_b{block_idx}_c0",
                    "source_file": state["pdf_name"],
                    "chunk_type": "text",
                    "source_page": text_block["start_page"],
                    "end_page": text_block["end_page"],
                    "chunk_index": 0,
                    "total_chunks_in_block": 1
                },
                "text_representation": content,
                "embedding_text": f"Documento: {state['pdf_name']}. {content}"
            }
            all_text_chunks.append(chunk)
            continue
        
        # Per blocchi più lunghi, usa AI per identificare punti di separazione semantica
        prompt = f"""Analizza il seguente testo e dividilo in chunk semanticamente coerenti per un sistema RAG.

TESTO DA ANALIZZARE:
{content}

REGOLE:
1. Ogni chunk deve essere autosufficiente e comprensibile senza contesto
2. Dimensione ideale: 500-800 caratteri (massimo 1000)
3. NON tagliare a metà frasi o paragrafi
4. Mantieni insieme concetti correlati
5. Identifica sezioni naturali (introduzioni, spiegazioni, conclusioni)

Rispondi SOLO con un array JSON di chunk, dove ogni chunk ha:
{{
  "content": "testo del chunk",
  "summary": "breve riassunto del contenuto (max 50 caratteri)",
  "key_concepts": ["concetto1", "concetto2"],
  "chunk_type": "introduction|explanation|data|conclusion|general"
}}"""

        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            generation_config = genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.3
            )
            
            response = model.generate_content(
                contents=prompt,
                generation_config=generation_config
            )
            
            if response.text:
                chunks_data = json.loads(response.text)
                
                # Processa ogni chunk identificato dall'AI
                for chunk_idx, chunk_data in enumerate(chunks_data):
                    chunk_content = chunk_data.get("content", "")
                    if not chunk_content.strip():
                        continue
                    
                    chunk = {
                        "content": chunk_content.strip(),
                        "metadata": {
                            "chunk_id": f"{state['pdf_name']}_text_b{block_idx}_c{chunk_idx}",
                            "source_file": state["pdf_name"],
                            "chunk_type": "text",
                            "semantic_type": chunk_data.get("chunk_type", "general"),
                            "source_page": text_block["start_page"],
                            "end_page": text_block["end_page"],
                            "chunk_index": chunk_idx,
                            "total_chunks_in_block": len(chunks_data),
                            "summary": chunk_data.get("summary", ""),
                            "key_concepts": chunk_data.get("key_concepts", [])
                        },
                        "text_representation": chunk_content.strip(),
                        "embedding_text": f"Documento: {state['pdf_name']}. {chunk_data.get('summary', '')}. {chunk_content.strip()}"
                    }
                    all_text_chunks.append(chunk)
                    
        except Exception as e:
            logger.error(f"Errore nel chunking AI, fallback a chunking semplice: {e}")
            # Fallback: chunking semplice basato su paragrafi
            paragraphs = content.split('\n\n')
            current_chunk = ""
            chunk_idx = 0
            
            for para in paragraphs:
                if len(current_chunk) + len(para) < 800:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk.strip():
                        chunk = {
                            "content": current_chunk.strip(),
                            "metadata": {
                                "chunk_id": f"{state['pdf_name']}_text_b{block_idx}_c{chunk_idx}",
                                "source_file": state["pdf_name"],
                                "chunk_type": "text",
                                "source_page": text_block["start_page"],
                                "end_page": text_block["end_page"],
                                "chunk_index": chunk_idx
                            },
                            "text_representation": current_chunk.strip(),
                            "embedding_text": f"Documento: {state['pdf_name']}. {current_chunk.strip()}"
                        }
                        all_text_chunks.append(chunk)
                        chunk_idx += 1
                    current_chunk = para + "\n\n"
            
            # Aggiungi l'ultimo chunk
            if current_chunk.strip():
                chunk = {
                    "content": current_chunk.strip(),
                    "metadata": {
                        "chunk_id": f"{state['pdf_name']}_text_b{block_idx}_c{chunk_idx}",
                        "source_file": state["pdf_name"],
                        "chunk_type": "text",
                        "source_page": text_block["start_page"],
                        "end_page": text_block["end_page"],
                        "chunk_index": chunk_idx
                    },
                    "text_representation": current_chunk.strip(),
                    "embedding_text": f"Documento: {state['pdf_name']}. {current_chunk.strip()}"
                }
                all_text_chunks.append(chunk)
    
    logger.info(f"    Generati {len(all_text_chunks)} chunk di testo con AI")
    return {"text_chunks": all_text_chunks}

def table_chunker_node(state: PDFChunkingState) -> Dict:
    """Invoca il tuo semantic_chunking_agent per ogni tabella."""
    logger.info(">>> NODO: Chunking Tabelle...")
    all_table_chunks = []
    
    for table in state.get("detected_tables", []):
        table_id = f"{state['pdf_name']}_p{table['page_num']}_t{table['table_index']}"
        
        agent_input = {
            "file_name": table_id,
            "rows": table["rows"],
            "columns": table["columns"],
            "extracted_cells": table["extracted_cells"],
            "regeneration_count": 0,
        }
        config = {"configurable": {"thread_id": f"table_chunker_{uuid.uuid4()}"}}
        
        # Chiamata al tuo agente esistente
        final_agent_state = chunking_agent.invoke(agent_input, config=config)
        
        agent_chunks = final_agent_state.get("chunks", [])
        for chunk in agent_chunks:
            # Normalizza la struttura per essere consistente con i text chunks
            formatted_chunk = {
                "content": chunk.get("content", {}),
                "metadata": {
                    **chunk.get("metadata", {}),
                    "source_page": table["page_num"],
                    "table_index": table["table_index"],
                    "chunk_type": "table",
                    "source_file": state["pdf_name"]
                },
                "text_representation": chunk.get("text_representation", ""),
                "embedding_text": chunk.get("embedding_text", "")
            }
            all_table_chunks.append(formatted_chunk)
    
    logger.info(f"    Generati {len(all_table_chunks)} chunk dalle tabelle")
    return {"table_chunks": all_table_chunks}

def final_merger_node(state: PDFChunkingState) -> Dict:
    """Unisce e formatta tutti i chunk per essere pronti all'upsert."""
    logger.info(">>> NODO: Preparazione finale chunks per upsert...")
    
    import hashlib
    from datetime import datetime
    
    text_chunks = state.get("text_chunks", [])
    table_chunks = state.get("table_chunks", [])
    
    logger.info(f"    Text chunks da unire: {len(text_chunks)}")
    logger.info(f"    Table chunks da unire: {len(table_chunks)}")
    
    final_chunks = []
    
    # Formatta ogni chunk per l'upsert nel vector DB
    all_chunks = text_chunks + table_chunks
    
    # Ordina per pagina
    all_chunks.sort(key=lambda x: (
        x["metadata"].get("source_page", 0),
        x["metadata"].get("table_index", -1),
        x["metadata"].get("chunk_index", 0)
    ))
    
    # Timestamp comune per questo batch
    batch_timestamp = datetime.now().isoformat()
    
    for idx, chunk in enumerate(all_chunks):
        # Genera ID deterministico basato sul contenuto per upsert idempotente
        content_str = str(chunk.get("content", ""))
        chunk_hash = hashlib.md5(
            f"{state['pdf_name']}_{content_str[:100]}_{idx}".encode()
        ).hexdigest()[:12]
        
        # Pulisci il contenuto se è testo con errori OCR evidenti
        content = chunk.get("content", "")
        if isinstance(content, str) and chunk["metadata"].get("chunk_type") == "text":
            # Rimuovi sequenze di numeri casuali tipiche di errori OCR
            content = re.sub(r'\b\d{1,2}\.\s*\d{2,}\s*\d{2,}', '', content)
            content = re.sub(r'\s+', ' ', content).strip()
        
        # Struttura pronta per upsert in un vector DB
        final_chunk = {
            # CAMPO CRITICO PER UPSERT
            "id": f"{state['pdf_name'].replace('.pdf', '')}_{chunk_hash}",
            
            # CONTENUTO
            "content": content,
            
            # METADATI COMPLETI
            "metadata": {
                **chunk.get("metadata", {}),
                "document_name": state["pdf_name"],
                "chunk_position": idx,
                "total_chunks": len(all_chunks),
                "batch_timestamp": batch_timestamp,
                "chunk_hash": chunk_hash
            },
            
            # CAMPI PER VECTOR DB
            "embedding_text": chunk.get("embedding_text", ""),  # Da vettorizzare
            "text_representation": chunk.get("text_representation", ""),  # Per display
            
            # CAMPI AGGIUNTIVI PER UPSERT
            "vector_db_fields": {
                "namespace": state["pdf_name"].replace(".pdf", ""),
                "score_boost": 1.0 if chunk["metadata"].get("chunk_type") == "table" else 0.8,
                "is_table": chunk["metadata"].get("chunk_type") == "table",
                "page_number": chunk["metadata"].get("source_page", 0)
            }
        }
        final_chunks.append(final_chunk)
    
    logger.info(f"    Chunks finali pronti per upsert: {len(final_chunks)}")
    
    # Log della distribuzione
    text_count = len([c for c in final_chunks if c["metadata"].get("chunk_type") == "text"])
    table_count = len([c for c in final_chunks if c["metadata"].get("chunk_type") == "table"])
    logger.info(f"    Distribuzione: {text_count} text, {table_count} table")
    
    return {"final_chunks": final_chunks}

# --- Grafo Orchestratore ---
def create_orchestrator_agent():
    workflow = StateGraph(PDFChunkingState)
    
    # Aggiungi tutti i nodi
    workflow.add_node("pdf_analyzer", pdf_analyzer_node)
    workflow.add_node("text_extractor", text_extractor_node)
    workflow.add_node("table_extractor", table_extractor_node)
    workflow.add_node("ai_text_chunker", ai_text_chunker_node)
    workflow.add_node("table_chunker", table_chunker_node)
    workflow.add_node("final_merger", final_merger_node)
    
    # Entry point
    workflow.set_entry_point("pdf_analyzer")

    # Routing basato sulla strategia
    def route_after_analysis(state: PDFChunkingState):
        if state.get("error_message"):
            return "end"
        strategy = state.get("processing_strategy", "no_content")
        logger.info(f"    Routing strategy: {strategy}")
        return strategy

    workflow.add_conditional_edges(
        "pdf_analyzer",
        route_after_analysis,
        {
            "text_only": "text_extractor",
            "tables_only": "table_extractor", 
            "mixed_content": "text_extractor",
            "no_content": END,
            "end": END
        }
    )
    
    # Se c'è solo testo
    workflow.add_edge("text_extractor", "ai_text_chunker")
    
    # Se ci sono tabelle dopo il testo (mixed)
    def route_after_text_chunking(state: PDFChunkingState):
        strategy = state.get("processing_strategy", "")
        if strategy == "mixed_content":
            return "process_tables"
        else:
            return "merge"
    
    workflow.add_conditional_edges(
        "ai_text_chunker",
        route_after_text_chunking,
        {
            "process_tables": "table_extractor",
            "merge": "final_merger"
        }
    )
    
    # Dopo estrazione tabelle
    workflow.add_edge("table_extractor", "table_chunker")
    
    # Se partito da tables_only
    def route_after_table_extraction(state: PDFChunkingState):
        strategy = state.get("processing_strategy", "")
        if strategy == "tables_only":
            return "table_chunker"
        else:  # mixed_content
            return "table_chunker"
    
    # Tutti i percorsi convergono al merger finale
    workflow.add_edge("table_chunker", "final_merger")
    workflow.add_edge("final_merger", END)
    
    return workflow.compile()

# Esponi una singola istanza compilata dell'agente orchestratore
orchestrator_agent = create_orchestrator_agent()