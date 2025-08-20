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
    strategy_validation: Optional[Dict[str, Any]]  # Nuovo campo per validazione
    regeneration_count: int  # Conta i tentativi di rigenerazione

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
    
    # Crea una rappresentazione testuale della tabella (limitata per il prompt)
    table_text = "Tabella estratta (prime 5 righe per analisi):\n"
    row_count = 0
    for row_idx in sorted(table_data.keys())[:6]:  # Prime 5 righe + header
        row_values = []
        for col_idx in sorted(table_data[row_idx].keys()):
            row_values.append(table_data[row_idx][col_idx])
        table_text += f"Riga {row_idx}: {' | '.join(row_values)}\n"
        row_count += 1
    
    if len(table_data) > 6:
        table_text += f"... (e altre {len(table_data) - 6} righe)\n"
    
    prompt = f"""Analizza la seguente tabella e identifica la sua struttura semantica.

{table_text}

IMPORTANTE: Determina se si tratta di:
1. Serie temporale (dati che cambiano nel tempo per entità fisse)
2. Tabella di azioni/task (lista di attività o compiti)
3. Tabella comparativa (confronto tra entità)
4. Tabella statistica aggregata (totali, medie, etc.)
5. Tabella descrittiva (informazioni generali)

Per le serie temporali:
- Se la prima riga contiene anni/date e la prima colonna contiene entità (città, prodotti, etc.), usa "row_based"
- Se la prima colonna contiene date/anni e le altre colonne sono metriche diverse, usa "column_based"

Rispondi SOLO con un oggetto JSON:
{{
    "headers": ["lista", "degli", "header"],
    "table_type": "tipo_tabella",  // temporal_series, actions, comparative, statistical, descriptive
    "primary_key_column": numero_colonna_chiave_primaria_o_null,
    "temporal_dimension": "rows" o "columns" o null,  // dove si trova la dimensione temporale
    "entity_dimension": "rows" o "columns" o null,  // dove si trovano le entità
    "semantic_structure": {{
        "description": "breve descrizione del contenuto",
        "key_concepts": ["concetti", "chiave"],
        "relationships": "relazione tra le colonne",
        "data_pattern": "pattern dei dati (es: serie storica per provincia)"
    }},
    "recommended_strategy": "row_based"  // IMPORTANTE: per serie temporali con entità nelle righe, usa row_based
}}"""

    try:
        import google.generativeai as genai
        import os
        
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-pro")
        
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
            
            # Estrai informazioni aggiuntive per la validazione
            recommended_strategy = schema.pop("recommended_strategy", "row_based")
            
            # Logica migliorata per serie temporali
            if schema.get("table_type") == "temporal_series":
                if schema.get("entity_dimension") == "rows":
                    recommended_strategy = "row_based"
                    logger.info("Serie temporale con entità nelle righe -> strategia row_based")
                elif schema.get("entity_dimension") == "columns":
                    recommended_strategy = "column_based"
                    logger.info("Serie temporale con entità nelle colonne -> strategia column_based")
            
            return {
                "table_schema": schema,
                "chunking_strategy": recommended_strategy,
                "error_message": None,
                "regeneration_count": 0
            }
        else:
            return {
                "error_message": "Nessuna risposta dall'analisi della struttura"
            }
            
    except Exception as e:
        logger.error(f"Errore nell'analisi della struttura: {e}")
        # Fallback con euristica migliorata
        headers = []
        if len(state["extracted_cells"]) > 0:
            for cell in state["extracted_cells"]:
                if cell.get("rowIndex") == 1:
                    headers.append(cell.get("value", ""))
        
        # Controlla se gli header sembrano anni
        years_in_headers = sum(1 for h in headers if h.isdigit() and 1900 < int(h) < 2100)
        
        fallback_schema = {
            "headers": headers if headers else ["Col1", "Col2", "Col3", "Col4", "Col5"],
            "table_type": "temporal_series" if years_in_headers > 3 else "descriptive",
            "primary_key_column": 1,
            "temporal_dimension": "columns" if years_in_headers > 3 else None,
            "entity_dimension": "rows" if years_in_headers > 3 else None,
            "semantic_structure": {
                "description": "Serie temporale" if years_in_headers > 3 else "Tabella con dati strutturati",
                "key_concepts": [],
                "relationships": "Dati correlati per riga",
                "data_pattern": "Serie storica per entità" if years_in_headers > 3 else "Dati generici"
            }
        }
        
        # Per serie temporali con entità nelle righe, usa row_based
        strategy = "row_based" if years_in_headers > 3 else "row_based"
        
        return {
            "table_schema": fallback_schema,
            "chunking_strategy": strategy,
            "error_message": None,
            "regeneration_count": 0
        }

def determine_chunking_strategy_node(state: ChunkingState) -> dict:
    """Determina o conferma la strategia di chunking ottimale"""
    logger.info("\n--- NODO: Determine Chunking Strategy ---")
    
    # Se abbiamo già una strategia dall'analisi, usiamola
    if state.get("chunking_strategy"):
        logger.info(f"Uso strategia consigliata: {state['chunking_strategy']}")
        return {}
    
    # Altrimenti, determina in base al tipo di tabella con logica migliorata
    table_schema = state.get("table_schema", {})
    table_type = table_schema.get("table_type", "descriptive")
    
    # Nuova logica per serie temporali
    if table_type == "temporal_series":
        # Per serie temporali, considera dove sono le entità
        if table_schema.get("entity_dimension") == "rows":
            strategy = "row_based"  # Ogni riga è un'entità con la sua serie temporale
        elif table_schema.get("entity_dimension") == "columns":
            strategy = "column_based"  # Ogni colonna è un'entità
        else:
            # Default per serie temporali: row_based
            strategy = "row_based"
    else:
        strategy_map = {
            "actions": "row_based",
            "comparative": "hybrid",
            "statistical": "row_based",  # Cambiato da column_based
            "descriptive": "row_based"
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
            
            # Estrai il nome dell'entità (primo valore della riga)
            entity_name = ""
            
            for col_idx in sorted(table_data[row_idx].keys()):
                value = table_data[row_idx][col_idx]
                if col_idx <= len(headers):
                    key = headers[col_idx - 1] if headers else f"Column_{col_idx}"
                    key = key.replace('\n', ' ').strip()
                    content[key] = value
                    
                    # Il primo valore è spesso il nome dell'entità
                    if col_idx == 1:
                        entity_name = value
                    
                    text_parts.append(f"{key}: {value}")
                    
                    # Estrai keywords dai valori
                    if len(value) > 3 and not value.replace(',', '').replace('.', '').isdigit():
                        keywords.extend([w for w in value.split() if len(w) > 4][:3])
            
            # Determina priorità se presente
            priority = None
            for key, val in content.items():
                if "priorit" in key.lower():
                    priority = val
                    break
            
            # Crea il chunk con metadati migliorati
            chunk_id = f"{state['file_name']}_row_{entity_name.replace(' ', '_') if entity_name else row_idx}"
            
            # Testo embedding migliorato per serie temporali
            if schema.get("table_type") == "temporal_series" and entity_name:
                embedding_text = f"Documento: {state['file_name']}. Dati per {entity_name}: {' '.join(text_parts)}"
            else:
                embedding_text = f"Documento: {state['file_name']}. {' '.join(text_parts)}"
            
            chunk = {
                "metadata": {
                    "chunk_id": chunk_id,
                    "source_file": state["file_name"],
                    "chunk_type": "table_row",
                    "entity_name": entity_name if entity_name else None,
                    "row_indices": [row_idx],
                    "column_indices": list(range(1, state["columns"] + 1)),
                    "priority": priority,
                    "keywords": list(set(keywords))[:10]
                },
                "content": content,
                "text_representation": " | ".join(text_parts),
                "embedding_text": embedding_text
            }
            
            chunks.append(chunk)
    
    elif strategy == "column_based":
        # Implementazione column_based con attenzione alle serie temporali
        for col_idx in range(1, state["columns"] + 1):
            column_values = []
            column_data = {}
            
            for row_idx in sorted(table_data.keys()):
                if row_idx in table_data and col_idx in table_data[row_idx]:
                    value = table_data[row_idx][col_idx]
                    column_values.append(value)
                    
                    # Per la prima colonna, usa i valori come chiavi
                    if col_idx == 1 or row_idx == 1:
                        continue
                    
                    # Usa il valore della prima colonna come chiave se disponibile
                    if 1 in table_data[row_idx]:
                        key = table_data[row_idx][1]
                        column_data[key] = value
            
            header = headers[col_idx - 1] if col_idx <= len(headers) else f"Column_{col_idx}"
            chunk_id = f"{state['file_name']}_col_{col_idx}"
            
            chunk = {
                "metadata": {
                    "chunk_id": chunk_id,
                    "source_file": state["file_name"],
                    "chunk_type": "table_column",
                    "column_name": header,
                    "row_indices": list(range(1, state["rows"] + 1)),
                    "column_indices": [col_idx],
                    "priority": None,
                    "keywords": []
                },
                "content": {
                    "column_name": header,
                    "values": column_values,
                    "data_map": column_data if column_data else None
                },
                "text_representation": f"{header}: {', '.join(column_values)}",
                "embedding_text": f"Documento: {state['file_name']}. Colonna {header}: {', '.join(column_values)}"
            }
            
            chunks.append(chunk)
    
    else:  # hybrid strategy
        # Raggruppa per priorità o altro criterio
        priority_groups = {}
        
        for row_idx in sorted(table_data.keys()):
            if row_idx == 1 and headers:  # Skip header
                continue
            
            # Trova la priorità o altro criterio di raggruppamento
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
        
        # Crea un chunk per ogni gruppo
        for priority, row_indices in priority_groups.items():
            content = {
                "group_key": priority,
                "items": []
            }
            text_parts = [f"Gruppo: {priority}"]
            keywords = [priority] if priority != "undefined" else []
            
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
                
                content["items"].append(row_content)
                text_parts.append(json.dumps(row_content, ensure_ascii=False))
            
            chunk_id = f"{state['file_name']}_group_{priority.replace(' ', '_')}"
            
            chunk = {
                "metadata": {
                    "chunk_id": chunk_id,
                    "source_file": state["file_name"],
                    "chunk_type": "hybrid_group",
                    "row_indices": row_indices,
                    "column_indices": list(range(1, state["columns"] + 1)),
                    "group_key": priority,
                    "keywords": list(set(keywords))[:15]
                },
                "content": content,
                "text_representation": " | ".join(text_parts),
                "embedding_text": f"Documento: {state['file_name']}. {' '.join(text_parts)}"
            }
            
            chunks.append(chunk)
    
    logger.info(f"Generati {len(chunks)} chunks con strategia {strategy}")
    
    return {
        "chunks": chunks,
        "error_message": None
    }

def validate_chunks_quality_node(state: ChunkingState) -> dict:
    """Nuovo nodo: Valida la qualità dei chunks generati usando LLM"""
    logger.info("\n--- NODO: Validate Chunks Quality (LLM) ---")
    
    # Se abbiamo già rigenerato una volta, accetta il risultato
    if state.get("regeneration_count", 0) >= 1:
        logger.info("Già rigenerato una volta, accetto il risultato corrente")
        return {"strategy_validation": {"is_optimal": True}}
    
    chunks = state.get("chunks", [])
    if not chunks:
        return {"error_message": "Nessun chunk da validare"}
    
    # Prepara un campione di chunks per la validazione
    sample_chunks = chunks[:3] if len(chunks) > 3 else chunks
    
    # Crea una rappresentazione testuale dei chunks
    chunks_text = "Esempio di chunks generati:\n\n"
    for i, chunk in enumerate(sample_chunks, 1):
        chunks_text += f"Chunk {i}:\n"
        chunks_text += f"  ID: {chunk['metadata']['chunk_id']}\n"
        chunks_text += f"  Tipo: {chunk['metadata']['chunk_type']}\n"
        chunks_text += f"  Contenuto (preview): {chunk['text_representation'][:200]}...\n\n"
    
    chunks_text += f"Totale chunks generati: {len(chunks)}\n"
    chunks_text += f"Strategia utilizzata: {state.get('chunking_strategy')}\n"
    
    # Informazioni sulla tabella
    table_info = f"""
Informazioni sulla tabella originale:
- Tipo: {state.get('table_schema', {}).get('table_type')}
- Headers: {state.get('table_schema', {}).get('headers', [])}
- Dimensioni: {state['rows']} righe x {state['columns']} colonne
- Pattern dati: {state.get('table_schema', {}).get('semantic_structure', {}).get('data_pattern', 'sconosciuto')}
"""
    
    prompt = f"""Sei un esperto di sistemi RAG (Retrieval Augmented Generation). 
Valuta se i chunks generati sono ottimali per rispondere a domande tipiche sui dati.

{table_info}

{chunks_text}

Considera questi scenari di query tipiche:
1. "Qual è l'andamento temporale per una specifica entità?" (es: "Andamento danni a Firenze nel tempo")
2. "Quali sono i valori per un anno specifico?" (es: "Danni nel 2019 per tutte le province")
3. "Confronto tra entità" (es: "Confronta Firenze e Arezzo")
4. "Aggregazioni e totali" (es: "Totale danni per anno")

Valuta:
- I chunks permettono di rispondere facilmente a queste query?
- Ogni chunk contiene informazioni coerenti e complete?
- La strategia di chunking è appropriata per il tipo di dati?

Se i dati sono serie temporali per entità (es: città/province nel tempo):
- row_based è OTTIMALE: ogni chunk contiene TUTTI i dati temporali per una singola entità
- column_based è SUBOTTIMALE: richiede recuperare molti chunks per ricostruire la serie di una singola entità

Rispondi SOLO con JSON:
{{
    "is_optimal": true/false,
    "reasoning": "spiegazione breve",
    "issues": ["lista", "dei", "problemi"] o [],
    "recommended_strategy": "row_based" o "column_based" o "hybrid" o null,
    "confidence": 0.0-1.0
}}"""

    try:
        import google.generativeai as genai
        import os
        
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-pro")
        
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.3,
            top_p=0.95
        )
        
        response = model.generate_content(
            contents=prompt,
            generation_config=generation_config
        )
        
        if response.text:
            validation = json.loads(response.text)
            logger.info(f"Validazione qualità: {validation}")
            
            return {
                "strategy_validation": validation
            }
        else:
            # Se non c'è risposta, assumiamo che sia ok
            return {
                "strategy_validation": {"is_optimal": True}
            }
            
    except Exception as e:
        logger.error(f"Errore nella validazione qualità: {e}")
        # In caso di errore, accettiamo i chunks correnti
        return {
            "strategy_validation": {"is_optimal": True}
        }

def validate_chunks_coverage_node(state: ChunkingState) -> dict:
    """Valida che tutti i dati siano stati inclusi nei chunks"""
    logger.info("\n--- NODO: Validate Chunks Coverage ---")
    
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

def check_quality_validation(state: ChunkingState) -> Literal["optimal", "regenerate"]:
    """Controlla se i chunks sono ottimali o necessitano rigenerazione"""
    validation = state.get("strategy_validation", {})
    
    if not validation.get("is_optimal", True):
        # Se non è ottimale e c'è una strategia raccomandata diversa
        recommended = validation.get("recommended_strategy")
        current = state.get("chunking_strategy")
        
        if recommended and recommended != current and state.get("regeneration_count", 0) < 1:
            logger.info(f"Chunks non ottimali. Strategia corrente: {current}, raccomandata: {recommended}")
            return "regenerate"
    
    return "optimal"

def check_chunks_validity(state: ChunkingState) -> Literal["valid", "invalid"]:
    if state.get("error_message") or not state.get("chunks"):
        logger.warning(f"Chunks non validi: {state.get('error_message')}")
        return "invalid"
    return "valid"

def apply_new_strategy_node(state: ChunkingState) -> dict:
    """Applica la nuova strategia raccomandata dalla validazione"""
    logger.info("\n--- NODO: Apply New Strategy ---")
    
    validation = state.get("strategy_validation", {})
    new_strategy = validation.get("recommended_strategy")
    
    if new_strategy:
        logger.info(f"Applicazione nuova strategia: {new_strategy}")
        logger.info(f"Motivo: {validation.get('reasoning', 'non specificato')}")
        
        return {
            "chunking_strategy": new_strategy,
            "regeneration_count": state.get("regeneration_count", 0) + 1
        }
    
    return {}

# --- Costruzione del Grafico ---
def create_chunking_agent():
    """Crea e compila l'agente di semantic chunking con validazione LLM"""
    checkpointer = InMemorySaver()
    workflow = StateGraph(ChunkingState)
    
    # Aggiungi i nodi
    workflow.add_node("analyze_structure", analyze_table_structure_node)
    workflow.add_node("determine_strategy", determine_chunking_strategy_node)
    workflow.add_node("generate_chunks", generate_chunks_node)
    workflow.add_node("validate_quality", validate_chunks_quality_node)  # Nuovo nodo
    workflow.add_node("apply_new_strategy", apply_new_strategy_node)  # Nuovo nodo
    workflow.add_node("validate_coverage", validate_chunks_coverage_node)
    
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
    workflow.add_edge("generate_chunks", "validate_quality")  # Prima valida la qualità
    
    workflow.add_conditional_edges(
        "validate_quality",
        check_quality_validation,
        {
            "optimal": "validate_coverage",
            "regenerate": "apply_new_strategy"
        }
    )
    
    workflow.add_edge("apply_new_strategy", "generate_chunks")  # Rigenera con nuova strategia
    
    workflow.add_conditional_edges(
        "validate_coverage",
        check_chunks_validity,
        {
            "valid": END,
            "invalid": END  # Per ora termina anche se invalido
        }
    )
    
    return workflow.compile(checkpointer=checkpointer)

# Esponi l'agente compilato
chunking_agent = create_chunking_agent()