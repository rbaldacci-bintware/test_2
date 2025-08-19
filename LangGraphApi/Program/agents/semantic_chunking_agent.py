# /agents/semantic_chunking_agent.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, List, Dict, Any, Literal, Optional
from services import gemini_service
from models.semantic_chunking_models import ChunkingStrategy, TableSchema, SemanticChunk, ChunkMetadata
import json
import uuid
import logging

# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Stato del Grafico ---
class ChunkingState(TypedDict):
    file_name: str
    rows: int
    columns: int
    extracted_cells: List[Dict[str, Any]]
    table_schema: Optional[Dict[str, Any]]
    chunking_strategy: Optional[str]
    chunks: List[Dict[str, Any]]
    error_message: Optional[str]

# --- Nodi del Grafico ---
def analyze_table_structure_node(state: ChunkingState) -> dict:
    """Analizza la struttura semantica della tabella usando LLM"""
    logger.info("\n--- NODO: Analyze Table Structure ---")
    
    # Ricostruisci la tabella in formato leggibile
    table_data = {}
    for cell in state["extracted_cells"]:
        row_idx = cell.get("rowIndex", 0)
        col_idx = cell.get("columnIndex", 0)
        if row_idx not in table_data:
            table_data[row_idx] = {}
        table_data[row_idx][col_idx] = cell.get("value", "")
    
    # Crea una rappresentazione testuale della tabella
    table_text = "Tabella estratta:\n"
    for row_idx in sorted(table_data.keys()):
        row_values = []
        for col_idx in sorted(table_data[row_idx].keys()):
            row_values.append(table_data[row_idx][col_idx])
        table_text += f"Riga {row_idx}: {' | '.join(row_values)}\n"
    
    prompt = f"""Analizza la seguente tabella e identifica la sua struttura semantica.

{table_text}

Rispondi SOLO con un oggetto JSON con questa struttura:
{{
    "headers": ["lista", "degli", "header"],
    "table_type": "tipo_tabella",  // Opzioni: "actions", "comparative", "statistical", "descriptive"
    "primary_key_column": numero_colonna_chiave_primaria_o_null,
    "semantic_structure": {{
        "description": "breve descrizione del contenuto",
        "key_concepts": ["concetti", "chiave"],
        "relationships": "relazione tra le colonne"
    }},
    "recommended_strategy": "row_based"  // o "column_based" o "hybrid"
}}

Analizza attentamente:
1. Se la prima riga contiene header/intestazioni
2. Il tipo di dati nella tabella
3. Come le informazioni sono correlate tra loro
4. La strategia di chunking più appropriata per un sistema RAG"""

    try:
        # Per questa analisi non abbiamo bisogno del PDF, solo del testo
        # Usiamo una chiamata semplificata a Gemini
        import google.generativeai as genai
        import os
        
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2,
            top_p=0.95
        )
        
        response = model.generate_content(
            contents=prompt,
            generation_config=generation_config
        )
        
        if response.text:
            schema = json.loads(response.text)
            logger.info(f"Schema identificato: {schema}")
            
            # Estrai la strategia consigliata e rimuovila dallo schema
            recommended_strategy = schema.pop("recommended_strategy", "row_based")
            
            return {
                "table_schema": schema,
                "chunking_strategy": recommended_strategy,
                "error_message": None
            }
        else:
            return {
                "error_message": "Nessuna risposta dall'analisi della struttura"
            }
            
    except Exception as e:
        logger.error(f"Errore nell'analisi della struttura: {e}")
        # Fallback: usa euristica semplice
        headers = []
        if len(state["extracted_cells"]) > 0:
            # Assume che la prima riga siano gli header
            for cell in state["extracted_cells"]:
                if cell.get("rowIndex") == 1:
                    headers.append(cell.get("value", ""))
        
        fallback_schema = {
            "headers": headers if headers else ["Col1", "Col2", "Col3", "Col4", "Col5"],
            "table_type": "descriptive",
            "primary_key_column": 1,
            "semantic_structure": {
                "description": "Tabella con dati strutturati",
                "key_concepts": [],
                "relationships": "Dati correlati per riga"
            }
        }
        
        return {
            "table_schema": fallback_schema,
            "chunking_strategy": "row_based",  # Default sicuro
            "error_message": None
        }

def determine_chunking_strategy_node(state: ChunkingState) -> dict:
    """Determina o conferma la strategia di chunking ottimale"""
    logger.info("\n--- NODO: Determine Chunking Strategy ---")
    
    # Se abbiamo già una strategia dall'analisi, usiamola
    if state.get("chunking_strategy"):
        logger.info(f"Uso strategia consigliata: {state['chunking_strategy']}")
        return {}
    
    # Altrimenti, determina in base al tipo di tabella
    table_type = state.get("table_schema", {}).get("table_type", "descriptive")
    
    strategy_map = {
        "actions": "row_based",      # Azioni -> chunk per riga
        "comparative": "hybrid",      # Comparazioni -> chunk ibridi
        "statistical": "column_based", # Statistiche -> potrebbe essere utile per colonna
        "descriptive": "row_based"    # Descrittivo -> chunk per riga
    }
    
    strategy = strategy_map.get(table_type, "row_based")
    logger.info(f"Strategia determinata per tipo '{table_type}': {strategy}")
    
    return {
        "chunking_strategy": strategy
    }

def generate_chunks_node(state: ChunkingState) -> dict:
    """Genera i chunk semantici basandosi sulla strategia scelta"""
    logger.info(f"\n--- NODO: Generate Chunks (Strategy: {state.get('chunking_strategy')}) ---")
    
    chunks = []
    strategy = state.get("chunking_strategy", "row_based")
    schema = state.get("table_schema", {})
    headers = schema.get("headers", [])
    
    # Ricostruisci la tabella
    table_data = {}
    for cell in state["extracted_cells"]:
        row_idx = cell.get("rowIndex", 0)
        col_idx = cell.get("columnIndex", 0)
        if row_idx not in table_data:
            table_data[row_idx] = {}
        table_data[row_idx][col_idx] = cell.get("value", "")
    
    if strategy == "row_based":
        # Genera un chunk per ogni riga (escludendo l'header)
        for row_idx in sorted(table_data.keys()):
            if row_idx == 1 and headers:  # Skip header row
                continue
            
            # Costruisci il contenuto del chunk
            content = {}
            text_parts = []
            keywords = []
            
            for col_idx in sorted(table_data[row_idx].keys()):
                value = table_data[row_idx][col_idx]
                # Usa l'header come chiave se disponibile
                if col_idx <= len(headers):
                    key = headers[col_idx - 1] if headers else f"Column_{col_idx}"
                    # Pulisci la chiave (rimuovi newline e spazi extra)
                    key = key.replace('\n', ' ').strip()
                    content[key] = value
                    text_parts.append(f"{key}: {value}")
                    
                    # Estrai keywords dai valori
                    if len(value) > 3:
                        keywords.extend([w for w in value.split() if len(w) > 4][:3])
            
            # Determina priorità se presente
            priority = None
            for key, val in content.items():
                if "priorit" in key.lower():
                    priority = val
                    break
            
            # Crea il chunk
            chunk_id = f"{state['file_name']}_row_{row_idx}"
            
            chunk = {
                "metadata": {
                    "chunk_id": chunk_id,
                    "source_file": state["file_name"],
                    "chunk_type": "table_row",
                    "row_indices": [row_idx],
                    "column_indices": list(range(1, state["columns"] + 1)),
                    "priority": priority,
                    "keywords": list(set(keywords))[:10]
                },
                "content": content,
                "text_representation": " | ".join(text_parts),
                "embedding_text": f"Documento: {state['file_name']}. {' '.join(text_parts)}"
            }
            
            chunks.append(chunk)
    
    elif strategy == "hybrid":
        # Raggruppa per priorità o altro criterio
        priority_groups = {}
        
        for row_idx in sorted(table_data.keys()):
            if row_idx == 1 and headers:  # Skip header
                continue
            
            # Trova la priorità
            priority = "undefined"
            for col_idx in sorted(table_data[row_idx].keys()):
                if col_idx <= len(headers):
                    header = headers[col_idx - 1] if headers else ""
                    if "priorit" in header.lower():
                        priority = table_data[row_idx][col_idx]
                        break
            
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(row_idx)
        
        # Crea un chunk per ogni gruppo di priorità
        for priority, row_indices in priority_groups.items():
            content = {
                "priority_level": priority,
                "actions": []
            }
            text_parts = [f"Gruppo di priorità: {priority}"]
            keywords = [priority]
            
            for row_idx in row_indices:
                row_content = {}
                for col_idx in sorted(table_data[row_idx].keys()):
                    value = table_data[row_idx][col_idx]
                    if col_idx <= len(headers):
                        key = headers[col_idx - 1] if headers else f"Column_{col_idx}"
                        key = key.replace('\n', ' ').strip()
                        row_content[key] = value
                        
                        if len(value) > 3:
                            keywords.extend([w for w in value.split() if len(w) > 4][:2])
                
                content["actions"].append(row_content)
                text_parts.append(json.dumps(row_content, ensure_ascii=False))
            
            chunk_id = f"{state['file_name']}_priority_{priority.replace(' ', '_')}"
            
            chunk = {
                "metadata": {
                    "chunk_id": chunk_id,
                    "source_file": state["file_name"],
                    "chunk_type": "priority_group",
                    "row_indices": row_indices,
                    "column_indices": list(range(1, state["columns"] + 1)),
                    "priority": priority,
                    "keywords": list(set(keywords))[:15]
                },
                "content": content,
                "text_representation": " | ".join(text_parts),
                "embedding_text": f"Documento: {state['file_name']}. {' '.join(text_parts)}"
            }
            
            chunks.append(chunk)
    
    else:  # column_based (raramente utile per tabelle di azioni)
        # Implementazione base per completezza
        for col_idx in range(1, state["columns"] + 1):
            column_values = []
            for row_idx in sorted(table_data.keys()):
                if row_idx in table_data and col_idx in table_data[row_idx]:
                    column_values.append(table_data[row_idx][col_idx])
            
            header = headers[col_idx - 1] if col_idx <= len(headers) else f"Column_{col_idx}"
            chunk_id = f"{state['file_name']}_col_{col_idx}"
            
            chunk = {
                "metadata": {
                    "chunk_id": chunk_id,
                    "source_file": state["file_name"],
                    "chunk_type": "table_column",
                    "row_indices": list(range(1, state["rows"] + 1)),
                    "column_indices": [col_idx],
                    "priority": None,
                    "keywords": []
                },
                "content": {
                    "column_name": header,
                    "values": column_values
                },
                "text_representation": f"{header}: {', '.join(column_values)}",
                "embedding_text": f"Documento: {state['file_name']}. Colonna {header}: {', '.join(column_values)}"
            }
            
            chunks.append(chunk)
    
    logger.info(f"Generati {len(chunks)} chunks con strategia {strategy}")
    
    return {
        "chunks": chunks,
        "error_message": None
    }

def validate_chunks_node(state: ChunkingState) -> dict:
    """Valida che tutti i dati siano stati inclusi nei chunks"""
    logger.info("\n--- NODO: Validate Chunks ---")
    
    total_cells = len(state["extracted_cells"])
    
    # Conta le celle coperte dai chunks
    covered_cells = set()
    for chunk in state.get("chunks", []):
        row_indices = chunk["metadata"].get("row_indices", [])
        col_indices = chunk["metadata"].get("column_indices", [])
        for row in row_indices:
            for col in col_indices:
                covered_cells.add((row, col))
    
    # Considera che potremmo escludere l'header row
    headers = state.get("table_schema", {}).get("headers", [])
    if headers:
        # Sottrai le celle dell'header dal totale
        total_cells -= state["columns"]
    
    coverage = len(covered_cells) / total_cells * 100 if total_cells > 0 else 0
    logger.info(f"Copertura dati: {coverage:.1f}% ({len(covered_cells)}/{total_cells} celle)")
    
    if coverage < 90:  # Soglia di copertura minima
        return {
            "error_message": f"Copertura dati insufficiente: {coverage:.1f}%"
        }
    
    return {
        "error_message": None
    }

# --- Funzioni di Routing ---
def check_structure_analysis(state: ChunkingState) -> Literal["success", "failed"]:
    if state.get("error_message") or not state.get("table_schema"):
        logger.warning("Analisi struttura fallita, uso fallback")
        return "failed"
    return "success"

def check_chunks_validity(state: ChunkingState) -> Literal["valid", "invalid"]:
    if state.get("error_message") or not state.get("chunks"):
        logger.warning(f"Chunks non validi: {state.get('error_message')}")
        return "invalid"
    return "valid"

# --- Costruzione del Grafico ---
def create_chunking_agent():
    """Crea e compila l'agente di semantic chunking"""
    checkpointer = InMemorySaver()
    workflow = StateGraph(ChunkingState)
    
    # Aggiungi i nodi
    workflow.add_node("analyze_structure", analyze_table_structure_node)
    workflow.add_node("determine_strategy", determine_chunking_strategy_node)
    workflow.add_node("generate_chunks", generate_chunks_node)
    workflow.add_node("validate_chunks", validate_chunks_node)
    
    # Definisci il flusso
    workflow.set_entry_point("analyze_structure")
    
    workflow.add_conditional_edges(
        "analyze_structure",
        check_structure_analysis,
        {
            "success": "determine_strategy",
            "failed": "determine_strategy"  # Procedi comunque con fallback
        }
    )
    
    workflow.add_edge("determine_strategy", "generate_chunks")
    workflow.add_edge("generate_chunks", "validate_chunks")
    
    workflow.add_conditional_edges(
        "validate_chunks",
        check_chunks_validity,
        {
            "valid": END,
            "invalid": END  # Per ora termina anche se invalido
        }
    )
    
    return workflow.compile(checkpointer=checkpointer)

# Esponi l'agente compilato
chunking_agent = create_chunking_agent()