# /Program/agents/hierarchical_chunking_agent.py
import io
import uuid
import logging
import json
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Any, Literal, TypedDict, Optional
from collections import defaultdict

# LangGraph
from langgraph.graph import StateGraph, END

# Utility per PDF
import fitz  # PyMuPDF
import pdfplumber

# Google AI per analisi semantica
import google.generativeai as genai
import os

# Importa agenti esistenti
from agents.semantic_chunking_agent import chunking_agent

# Configurazione del logger
logger = logging.getLogger(__name__)

# Configura Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Stato del Grafo ---
class HierarchicalChunkingState(TypedDict):
    pdf_file: bytes
    pdf_name: str
    pdf_metadata: Dict[str, Any]
    
    # Struttura gerarchica
    document_structure: Dict[str, Any]  # Struttura gerarchica completa
    hierarchy_elements: List[Dict]  # Lista di tutti gli elementi con marcatori univoci
    
    # Contenuti estratti
    text_blocks: List[Dict]  # Blocchi di testo con gerarchia
    tables_info: List[Dict]  # Info tabelle con gerarchia
    
    # Chunks finali
    text_chunks: List[Dict]
    table_chunks: List[Dict]
    final_chunks: List[Dict]
    
    # Gestione errori
    error_message: Optional[str]
    special_cases: List[Dict]  # Casi speciali da gestire

# --- Nodi del Grafo ---

def analyze_document_structure_node(state: HierarchicalChunkingState) -> Dict:
    """Analizza la struttura gerarchica del documento usando Gemini."""
    logger.info(">>> NODO: Analisi Struttura Gerarchica del Documento...")
    
    try:
        # Estrai prime pagine per analisi struttura
        doc = fitz.open(stream=state["pdf_file"], filetype="pdf")
        
        # Prepara testo campione per analisi (prime 5 pagine)
        sample_text = ""
        for i, page in enumerate(doc[:min(5, len(doc))]):
            page_text = page.get_text()
            sample_text += f"\n--- PAGINA {i+1} ---\n{page_text}\n"
        
        prompt = f"""Analizza la struttura gerarchica di questo documento PDF.
        
TESTO CAMPIONE:
{sample_text[:8000]}

COMPITI:
1. Identifica TUTTI i livelli gerarchici presenti (titolo documento, capitoli, paragrafi, sottoparagrafi, sotto-sottoparagrafi, etc.)
2. Riconosci i pattern di numerazione (1., 1.1, 1.1.1, oppure I, II, III, oppure lettere, etc.)
3. Identifica eventuali sezioni speciali (abstract, introduzione, conclusioni, appendici, bibliografia)
4. Nota la presenza di tabelle e figure con i loro riferimenti

IMPORTANTE: Devi identificare la struttura COMPLETA del documento, non solo il campione.
Se il documento continua oltre il campione, inferisci la struttura basandoti sui pattern identificati.

Rispondi SOLO con JSON:
{{
    "document_title": "titolo completo del documento",
    "structure_type": "numbered|named|mixed",  // tipo di struttura
    "hierarchy_patterns": {{
        "level_1": "pattern (es: numeri romani, numeri arabi)",
        "level_2": "pattern (es: 1.1, 1.2)",
        "level_3": "pattern (es: 1.1.1, lettere)",
        "level_4": "pattern se presente"
    }},
    "identified_sections": [
        {{
            "level": 1,  // livello gerarchico
            "marker": "1" o "I" o "Capitolo 1",  // marcatore originale
            "title": "Titolo della sezione",
            "type": "chapter|paragraph|subparagraph|section",
            "page_start": 1,  // pagina stimata di inizio
            "has_subsections": true/false
        }}
    ],
    "special_sections": [
        {{
            "type": "abstract|introduction|conclusion|appendix|bibliography",
            "title": "nome sezione",
            "page": 1
        }}
    ],
    "tables_detected": true/false,
    "figures_detected": true/false,
    "structure_confidence": 0.0-1.0
}}"""

        model = genai.GenerativeModel("gemini-2.5-pro")
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2
        )
        
        response = model.generate_content(
            contents=prompt,
            generation_config=generation_config
        )
        
        if response.text:
            structure = json.loads(response.text)
            logger.info(f"Struttura documento identificata: {structure['document_title']}")
            logger.info(f"Tipo struttura: {structure['structure_type']}")
            logger.info(f"Sezioni identificate: {len(structure.get('identified_sections', []))}")
            
            return {
                "document_structure": structure,
                "error_message": None
            }
        else:
            return {"error_message": "Nessuna risposta dall'analisi struttura"}
            
    except Exception as e:
        logger.error(f"Errore nell'analisi struttura: {e}")
        return {"error_message": f"Errore analisi struttura: {e}"}

def mark_hierarchy_elements_node(state: HierarchicalChunkingState) -> Dict:
    """Crea marcatori univoci per ogni elemento gerarchico."""
    logger.info(">>> NODO: Marcatura Elementi Gerarchici...")
    
    structure = state.get("document_structure", {})
    sections = structure.get("identified_sections", [])
    
    hierarchy_elements = []
    doc_id = hashlib.md5(state["pdf_name"].encode()).hexdigest()[:8]
    
    # Contatori per ogni livello
    level_counters = defaultdict(int)
    
    for section in sections:
        level = section["level"]
        level_counters[level] += 1
        
        # Reset contatori livelli inferiori
        for l in range(level + 1, 5):
            level_counters[l] = 0
        
        # Crea marcatore univoco basato sulla gerarchia
        marker_parts = [f"DOC_{doc_id}"]
        
        if level >= 1 and level_counters[1] > 0:
            marker_parts.append(f"CAP_{level_counters[1]:02d}")
        if level >= 2 and level_counters[2] > 0:
            marker_parts.append(f"PAR_{level_counters[2]:02d}")
        if level >= 3 and level_counters[3] > 0:
            marker_parts.append(f"SUB_{level_counters[3]:02d}")
        if level >= 4 and level_counters[4] > 0:
            marker_parts.append(f"SUBSUB_{level_counters[4]:02d}")
        
        unique_marker = "_".join(marker_parts)
        
        element = {
            "unique_id": unique_marker,
            "level": level,
            "original_marker": section.get("marker", ""),
            "title": section.get("title", ""),
            "type": section.get("type", ""),
            "page_start": section.get("page_start", 1),
            "parent_id": "_".join(marker_parts[:-1]) if len(marker_parts) > 1 else None,
            "children_ids": []  # Sarà popolato successivamente
        }
        
        hierarchy_elements.append(element)
    
    # Collega parent-children relationships
    for element in hierarchy_elements:
        if element["parent_id"]:
            parent = next((e for e in hierarchy_elements if e["unique_id"] == element["parent_id"]), None)
            if parent:
                parent["children_ids"].append(element["unique_id"])
    
    logger.info(f"Creati {len(hierarchy_elements)} marcatori univoci")
    for elem in hierarchy_elements[:5]:  # Log primi 5 elementi
        logger.info(f"  {elem['unique_id']}: {elem['title']}")
    
    return {"hierarchy_elements": hierarchy_elements}

def extract_hierarchical_text_node(state: HierarchicalChunkingState) -> Dict:
    """Estrae il testo mantenendo i riferimenti gerarchici."""
    logger.info(">>> NODO: Estrazione Testo con Gerarchia...")
    
    doc = fitz.open(stream=state["pdf_file"], filetype="pdf")
    hierarchy_elements = state.get("hierarchy_elements", [])
    
    # Crea mappa per ricerca rapida
    hierarchy_map = {elem["unique_id"]: elem for elem in hierarchy_elements}
    
    # Estrai tutto il testo del documento
    full_text = ""
    page_boundaries = []
    
    for i, page in enumerate(doc):
        page_start = len(full_text)
        page_text = page.get_text()
        full_text += page_text
        page_boundaries.append((i + 1, page_start, len(full_text)))
    
    # Usa Gemini per mappare il testo agli elementi gerarchici
    prompt = f"""Analizza questo testo e associa ogni sezione al suo elemento gerarchico.

ELEMENTI GERARCHICI IDENTIFICATI:
{json.dumps([{{
    "id": elem["unique_id"],
    "marker": elem["original_marker"],
    "title": elem["title"]
}} for elem in hierarchy_elements[:20]], indent=2)}

TESTO COMPLETO (prime 10000 caratteri):
{full_text[:10000]}

Per ogni sezione di testo, identifica:
1. A quale elemento gerarchico appartiene (usa l'ID univoco)
2. Il testo esatto della sezione
3. La posizione approssimativa (carattere di inizio/fine)

Rispondi SOLO con JSON:
{{
    "text_blocks": [
        {{
            "hierarchy_id": "DOC_xxx_CAP_01_PAR_01",
            "content": "testo della sezione",
            "start_char": 0,
            "end_char": 500,
            "contains_special_elements": ["table", "figure", "equation"] o []
        }}
    ],
    "unmapped_text": [  // testo che non appartiene a nessuna sezione
        {{
            "content": "testo non mappato",
            "suggested_parent": "ID del probabile parent",
            "reason": "motivo per cui non è mappato"
        }}
    ]
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
            mapping = json.loads(response.text)
            text_blocks = []
            
            for block in mapping.get("text_blocks", []):
                # Trova la pagina del blocco
                page_num = 1
                start_char = block.get("start_char", 0)
                for page, start, end in page_boundaries:
                    if start <= start_char < end:
                        page_num = page
                        break
                
                # Arricchisci con info gerarchiche complete
                hierarchy_id = block.get("hierarchy_id")
                hierarchy_info = hierarchy_map.get(hierarchy_id, {})
                
                # Costruisci la gerarchia completa
                full_hierarchy = {}
                current_id = hierarchy_id
                level = hierarchy_info.get("level", 0)
                
                # Risali la gerarchia
                while current_id and current_id in hierarchy_map:
                    elem = hierarchy_map[current_id]
                    level_name = {
                        1: "chapter",
                        2: "paragraph", 
                        3: "subparagraph",
                        4: "subsubparagraph"
                    }.get(elem["level"], f"level_{elem['level']}")
                    
                    full_hierarchy[level_name] = elem["title"]
                    current_id = elem.get("parent_id")
                
                text_blocks.append({
                    "unique_id": f"{hierarchy_id}_BLOCK_{len(text_blocks):03d}",
                    "hierarchy_id": hierarchy_id,
                    "hierarchy": full_hierarchy,
                    "content": block.get("content", ""),
                    "page_number": page_num,
                    "special_elements": block.get("contains_special_elements", [])
                })
            
            # Gestisci testo non mappato
            special_cases = []
            for unmapped in mapping.get("unmapped_text", []):
                special_cases.append({
                    "type": "unmapped_text",
                    "content": unmapped.get("content", ""),
                    "suggested_parent": unmapped.get("suggested_parent"),
                    "reason": unmapped.get("reason", "")
                })
            
            logger.info(f"Estratti {len(text_blocks)} blocchi di testo gerarchici")
            logger.info(f"Trovati {len(special_cases)} casi speciali")
            
            return {
                "text_blocks": text_blocks,
                "special_cases": special_cases
            }
            
    except Exception as e:
        logger.error(f"Errore nell'estrazione gerarchica: {e}")
        # Fallback: estrazione semplice
        return extract_simple_text_blocks(doc, hierarchy_elements)

def extract_simple_text_blocks(doc, hierarchy_elements):
    """Fallback per estrazione testo semplice."""
    text_blocks = []
    
    for i, page in enumerate(doc):
        page_text = page.get_text().strip()
        if page_text:
            text_blocks.append({
                "unique_id": f"PAGE_{i+1:03d}_BLOCK_001",
                "hierarchy_id": None,
                "hierarchy": {"page": f"Pagina {i+1}"},
                "content": page_text,
                "page_number": i + 1,
                "special_elements": []
            })
    
    return {"text_blocks": text_blocks, "special_cases": []}

def identify_tables_with_hierarchy_node(state: HierarchicalChunkingState) -> Dict:
    """Identifica le tabelle e la loro posizione gerarchica."""
    logger.info(">>> NODO: Identificazione Tabelle con Gerarchia...")
    
    tables_info = []
    text_blocks = state.get("text_blocks", [])
    
    with pdfplumber.open(io.BytesIO(state["pdf_file"])) as pdf:
        for i, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            
            if page_tables:
                # Trova il blocco di testo corrispondente alla pagina
                page_blocks = [b for b in text_blocks if b["page_number"] == i + 1]
                
                for j, table in enumerate(page_tables):
                    if not table or len(table) == 0:
                        continue
                    
                    # Determina la gerarchia della tabella
                    hierarchy = {}
                    hierarchy_id = None
                    
                    if page_blocks:
                        # Prendi la gerarchia del primo blocco della pagina
                        hierarchy = page_blocks[0].get("hierarchy", {})
                        hierarchy_id = page_blocks[0].get("hierarchy_id")
                    
                    # Converti tabella in celle per semantic_chunking_agent
                    cells = []
                    for row_idx, row in enumerate(table):
                        for col_idx, cell_value in enumerate(row):
                            cells.append({
                                "rowIndex": row_idx + 1,
                                "columnIndex": col_idx + 1,
                                "value": str(cell_value or "")
                            })
                    
                    tables_info.append({
                        "unique_id": f"TABLE_P{i+1:03d}_T{j+1:02d}",
                        "hierarchy_id": hierarchy_id,
                        "hierarchy": hierarchy,
                        "page_number": i + 1,
                        "table_index": j,
                        "rows": len(table),
                        "columns": len(table[0]) if table else 0,
                        "extracted_cells": cells
                    })
    
    logger.info(f"Identificate {len(tables_info)} tabelle con contesto gerarchico")
    
    # Se ci sono tabelle multi-pagina o casi speciali, usa Gemini
    if len(tables_info) > 0:
        tables_info = resolve_table_special_cases(state, tables_info)
    
    return {"tables_info": tables_info}

def resolve_table_special_cases(state: HierarchicalChunkingState, tables_info: List[Dict]) -> List[Dict]:
    """Risolve casi speciali delle tabelle con Gemini."""
    
    # Identifica potenziali tabelle multi-pagina
    consecutive_tables = []
    for i in range(len(tables_info) - 1):
        current = tables_info[i]
        next_table = tables_info[i + 1]
        
        # Se sono su pagine consecutive e hanno stesso numero di colonne
        if (next_table["page_number"] == current["page_number"] + 1 and
            next_table["columns"] == current["columns"]):
            consecutive_tables.append((i, i + 1))
    
    if consecutive_tables:
        prompt = f"""Analizza queste tabelle consecutive e determina se sono parti della stessa tabella:

{json.dumps([{
    "page": t["page_number"],
    "rows": t["rows"],
    "columns": t["columns"],
    "first_row": t["extracted_cells"][:t["columns"]] if t["extracted_cells"] else []
} for t in tables_info], indent=2)}

Rispondi SOLO con JSON:
{{
    "merged_tables": [
        {{
            "table_indices": [0, 1],  // indici delle tabelle da unire
            "reason": "continuazione della stessa tabella"
        }}
    ]
}}"""

        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            generation_config = genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
            
            response = model.generate_content(
                contents=prompt,
                generation_config=generation_config
            )
            
            if response.text:
                result = json.loads(response.text)
                
                # Unisci le tabelle identificate
                for merge_info in result.get("merged_tables", []):
                    indices = merge_info["table_indices"]
                    if len(indices) > 1:
                        # Unisci le celle
                        merged_cells = []
                        for idx in indices:
                            if idx < len(tables_info):
                                merged_cells.extend(tables_info[idx]["extracted_cells"])
                        
                        # Aggiorna la prima tabella
                        tables_info[indices[0]]["extracted_cells"] = merged_cells
                        tables_info[indices[0]]["rows"] = sum(tables_info[idx]["rows"] for idx in indices)
                        tables_info[indices[0]]["is_multipage"] = True
                        
                        # Marca le altre per rimozione
                        for idx in indices[1:]:
                            tables_info[idx]["merged_into"] = indices[0]
                
                # Rimuovi tabelle unite
                tables_info = [t for t in tables_info if "merged_into" not in t]
                
        except Exception as e:
            logger.error(f"Errore nella risoluzione casi speciali tabelle: {e}")
    
    return tables_info

def create_hierarchical_text_chunks_node(state: HierarchicalChunkingState) -> Dict:
    """Crea chunk di testo mantenendo la gerarchia completa."""
    logger.info(">>> NODO: Creazione Chunk Gerarchici di Testo...")
    
    text_blocks = state.get("text_blocks", [])
    text_chunks = []
    
    for block in text_blocks:
        content = block.get("content", "")
        
        # Se il blocco è piccolo, un chunk singolo
        if len(content) < 1000:
            chunk = {
                "content": content,
                "metadata": {
                    "chunk_id": f"{block['unique_id']}_CHUNK_001",
                    "source_file": state["pdf_name"],
                    "chunk_type": "text",
                    "hierarchy_id": block.get("hierarchy_id"),
                    "hierarchy": block.get("hierarchy", {}),
                    "page_number": block.get("page_number", 1),
                    "has_special_elements": len(block.get("special_elements", [])) > 0,
                    "special_elements": block.get("special_elements", [])
                },
                "text_representation": content,
                "embedding_text": create_embedding_text(
                    content, 
                    block.get("hierarchy", {}), 
                    state["pdf_name"]
                )
            }
            text_chunks.append(chunk)
        else:
            # Per blocchi grandi, dividi mantenendo il contesto
            chunks = split_text_with_hierarchy(
                content, 
                block, 
                state["pdf_name"],
                max_chunk_size=800
            )
            text_chunks.extend(chunks)
    
    logger.info(f"Creati {len(text_chunks)} chunk di testo gerarchici")
    
    return {"text_chunks": text_chunks}

def split_text_with_hierarchy(content: str, block: Dict, pdf_name: str, max_chunk_size: int) -> List[Dict]:
    """Divide il testo in chunk mantenendo la gerarchia."""
    chunks = []
    
    # Dividi per paragrafi
    paragraphs = content.split('\n\n')
    current_chunk = ""
    chunk_index = 0
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk.strip():
                chunk_index += 1
                chunks.append({
                    "content": current_chunk.strip(),
                    "metadata": {
                        "chunk_id": f"{block['unique_id']}_CHUNK_{chunk_index:03d}",
                        "source_file": pdf_name,
                        "chunk_type": "text",
                        "hierarchy_id": block.get("hierarchy_id"),
                        "hierarchy": block.get("hierarchy", {}),
                        "page_number": block.get("page_number", 1),
                        "chunk_index": chunk_index,
                        "total_chunks_in_block": None  # Sarà aggiornato dopo
                    },
                    "text_representation": current_chunk.strip(),
                    "embedding_text": create_embedding_text(
                        current_chunk.strip(),
                        block.get("hierarchy", {}),
                        pdf_name
                    )
                })
            current_chunk = para + "\n\n"
    
    # Ultimo chunk
    if current_chunk.strip():
        chunk_index += 1
        chunks.append({
            "content": current_chunk.strip(),
            "metadata": {
                "chunk_id": f"{block['unique_id']}_CHUNK_{chunk_index:03d}",
                "source_file": pdf_name,
                "chunk_type": "text",
                "hierarchy_id": block.get("hierarchy_id"),
                "hierarchy": block.get("hierarchy", {}),
                "page_number": block.get("page_number", 1),
                "chunk_index": chunk_index,
                "total_chunks_in_block": len(chunks) + 1
            },
            "text_representation": current_chunk.strip(),
            "embedding_text": create_embedding_text(
                current_chunk.strip(),
                block.get("hierarchy", {}),
                pdf_name
            )
        })
    
    # Aggiorna total_chunks_in_block
    for chunk in chunks:
        chunk["metadata"]["total_chunks_in_block"] = len(chunks)
    
    return chunks

def create_embedding_text(content: str, hierarchy: Dict, pdf_name: str) -> str:
    """Crea testo ottimizzato per embedding con contesto gerarchico."""
    hierarchy_path = " > ".join([
        hierarchy.get("chapter", ""),
        hierarchy.get("paragraph", ""),
        hierarchy.get("subparagraph", "")
    ]).strip(" > ")
    
    if hierarchy_path:
        return f"Documento: {pdf_name}. Sezione: {hierarchy_path}. Contenuto: {content[:500]}"
    else:
        return f"Documento: {pdf_name}. {content[:500]}"

def process_tables_with_hierarchy_node(state: HierarchicalChunkingState) -> Dict:
    """Processa le tabelle usando semantic_chunking_agent mantenendo la gerarchia."""
    logger.info(">>> NODO: Elaborazione Tabelle con Gerarchia...")
    
    tables_info = state.get("tables_info", [])
    table_chunks = []
    
    for table in tables_info:
        # Usa semantic_chunking_agent per il chunking
        table_id = table["unique_id"]
        
        agent_input = {
            "file_name": table_id,
            "rows": table["rows"],
            "columns": table["columns"],
            "extracted_cells": table["extracted_cells"],
            "regeneration_count": 0,
        }
        
        config = {"configurable": {"thread_id": f"table_{uuid.uuid4()}"}}
        
        # Invoca l'agente esistente
        try:
            final_state = chunking_agent.invoke(agent_input, config=config)
            
            # Arricchisci i chunk con info gerarchiche
            for chunk in final_state.get("chunks", []):
                enriched_chunk = {
                    **chunk,
                    "metadata": {
                        **chunk.get("metadata", {}),
                        "hierarchy_id": table["hierarchy_id"],
                        "hierarchy": table["hierarchy"],
                        "page_number": table["page_number"],
                        "chunk_type": "table",
                        "source_file": state["pdf_name"]
                    },
                    "embedding_text": create_embedding_text(
                        chunk.get("text_representation", ""),
                        table["hierarchy"],
                        state["pdf_name"]
                    )
                }
                table_chunks.append(enriched_chunk)
                
        except Exception as e:
            logger.error(f"Errore nel processing tabella {table_id}: {e}")
            # Crea chunk fallback
            table_chunks.append({
                "content": {"error": str(e), "cells": table["extracted_cells"][:10]},
                "metadata": {
                    "chunk_id": f"{table_id}_FALLBACK",
                    "hierarchy_id": table["hierarchy_id"],
                    "hierarchy": table["hierarchy"],
                    "page_number": table["page_number"],
                    "chunk_type": "table",
                    "source_file": state["pdf_name"]
                },
                "text_representation": f"Tabella su pagina {table['page_number']}",
                "embedding_text": f"Documento: {state['pdf_name']}. Tabella non processata."
            })
    
    logger.info(f"Creati {len(table_chunks)} chunk da tabelle")
    
    return {"table_chunks": table_chunks}

def handle_special_cases_node(state: HierarchicalChunkingState) -> Dict:
    """Gestisce casi speciali con l'aiuto di Gemini."""
    logger.info(">>> NODO: Gestione Casi Speciali...")
    
    special_cases = state.get("special_cases", [])
    
    if not special_cases:
        logger.info("Nessun caso speciale da gestire")
        return {}
    
    # Usa Gemini per risolvere i casi speciali
    prompt = f"""Risolvi questi casi speciali di contenuto non strutturato:

{json.dumps(special_cases, indent=2)}

Per ogni caso, suggerisci:
1. Come gestirlo (includere/escludere/assegnare a parent)
2. Se includere, a quale elemento gerarchico assegnarlo

Rispondi SOLO con JSON:
{{
    "resolutions": [
        {{
            "case_index": 0,
            "action": "include|exclude|assign_to_parent",
            "parent_id": "ID gerarchico se action è assign_to_parent",
            "reason": "motivazione"
        }}
    ]
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
            resolutions = json.loads(response.text)
            
            # Applica le risoluzioni
            additional_chunks = []
            for resolution in resolutions.get("resolutions", []):
                idx = resolution["case_index"]
                if idx < len(special_cases):
                    case = special_cases[idx]
                    
                    if resolution["action"] == "include" or resolution["action"] == "assign_to_parent":
                        # Crea chunk per il caso speciale
                        chunk = {
                            "content": case.get("content", ""),
                            "metadata": {
                                "chunk_id": f"SPECIAL_{idx:03d}",
                                "source_file": state["pdf_name"],
                                "chunk_type": "text_special",
                                "hierarchy_id": resolution.get("parent_id"),
                                "hierarchy": {"special": "Contenuto non strutturato"},
                                "special_case_type": case.get("type", "unknown"),
                                "resolution_reason": resolution.get("reason", "")
                            },
                            "text_representation": case.get("content", ""),
                            "embedding_text": f"Documento: {state['pdf_name']}. Contenuto speciale: {case.get('content', '')[:200]}"
                        }
                        additional_chunks.append(chunk)
            
            # Aggiungi i chunk speciali ai text_chunks
            text_chunks = state.get("text_chunks", [])
            text_chunks.extend(additional_chunks)
            
            logger.info(f"Risolti {len(resolutions.get('resolutions', []))} casi speciali")
            logger.info(f"Aggiunti {len(additional_chunks)} chunk speciali")
            
            return {"text_chunks": text_chunks}
            
    except Exception as e:
        logger.error(f"Errore nella gestione casi speciali: {e}")
    
    return {}

def merge_and_finalize_chunks_node(state: HierarchicalChunkingState) -> Dict:
    """Unisce e finalizza tutti i chunk con metadati completi."""
    logger.info(">>> NODO: Finalizzazione Chunk Gerarchici...")
    
    text_chunks = state.get("text_chunks", [])
    table_chunks = state.get("table_chunks", [])
    
    final_chunks = []
    batch_timestamp = datetime.now().isoformat()
    
    # Combina tutti i chunk
    all_chunks = text_chunks + table_chunks
    
    # Ordina per gerarchia e poi per pagina
    all_chunks.sort(key=lambda x: (
        x["metadata"].get("hierarchy_id", "ZZZ"),
        x["metadata"].get("page_number", 0),
        x["metadata"].get("chunk_index", 0)
    ))
    
    # Prepara per upsert
    for idx, chunk in enumerate(all_chunks):
        # Genera ID deterministico per upsert
        content_str = str(chunk.get("content", ""))[:100]
        chunk_hash = hashlib.md5(
            f"{state['pdf_name']}_{content_str}_{idx}".encode()
        ).hexdigest()[:12]
        
        final_chunk = {
            "id": f"{state['pdf_name'].replace('.pdf', '')}_{chunk_hash}",
            "content": chunk.get("content", ""),
            "metadata": {
                **chunk.get("metadata", {}),
                "document_name": state["pdf_name"],
                "chunk_position": idx,
                "total_chunks": len(all_chunks),
                "batch_timestamp": batch_timestamp,
                "chunk_hash": chunk_hash,
                "has_hierarchy": bool(chunk["metadata"].get("hierarchy_id"))
            },
            "embedding_text": chunk.get("embedding_text", ""),
            "text_representation": chunk.get("text_representation", ""),
            "hierarchy": chunk["metadata"].get("hierarchy", {}),
            "vector_db_fields": {
                "namespace": state["pdf_name"].replace(".pdf", ""),
                "score_boost": 1.0 if chunk["metadata"].get("chunk_type") == "table" else 0.9,
                "is_table": chunk["metadata"].get("chunk_type") == "table",
                "page_number": chunk["metadata"].get("page_number", 0),
                "hierarchy_level": len(chunk["metadata"].get("hierarchy", {}))
            }
        }
        final_chunks.append(final_chunk)
    
    # Log statistiche
    logger.info(f">>> Chunk finali pronti: {len(final_chunks)}")
    text_count = len([c for c in final_chunks if c["metadata"].get("chunk_type") == "text"])
    table_count = len([c for c in final_chunks if c["metadata"].get("chunk_type") == "table"])
    special_count = len([c for c in final_chunks if c["metadata"].get("chunk_type") == "text_special"])
    
    logger.info(f"  - Text chunks: {text_count}")
    logger.info(f"  - Table chunks: {table_count}")
    logger.info(f"  - Special chunks: {special_count}")
    
    # Log esempi di gerarchia
    for chunk in final_chunks[:3]:
        hierarchy = chunk.get("hierarchy", {})
        if hierarchy:
            path = " > ".join(hierarchy.values())
            logger.info(f"  Esempio gerarchia: {path}")
    
    return {"final_chunks": final_chunks}

# --- Grafo Orchestratore ---
def create_hierarchical_chunking_agent():
    """Crea l'agente di chunking gerarchico."""
    workflow = StateGraph(HierarchicalChunkingState)
    
    # Aggiungi i nodi
    workflow.add_node("analyze_structure", analyze_document_structure_node)
    workflow.add_node("mark_elements", mark_hierarchy_elements_node)
    workflow.add_node("extract_text", extract_hierarchical_text_node)
    workflow.add_node("identify_tables", identify_tables_with_hierarchy_node)
    workflow.add_node("create_text_chunks", create_hierarchical_text_chunks_node)
    workflow.add_node("process_tables", process_tables_with_hierarchy_node)
    workflow.add_node("handle_special_cases", handle_special_cases_node)
    workflow.add_node("finalize", merge_and_finalize_chunks_node)
    
    # Definisci il flusso
    workflow.set_entry_point("analyze_structure")
    
    # Flusso sequenziale principale
    workflow.add_edge("analyze_structure", "mark_elements")
    workflow.add_edge("mark_elements", "extract_text")
    workflow.add_edge("extract_text", "identify_tables")
    workflow.add_edge("identify_tables", "create_text_chunks")
    workflow.add_edge("create_text_chunks", "process_tables")
    workflow.add_edge("process_tables", "handle_special_cases")
    workflow.add_edge("handle_special_cases", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow.compile()

# Esponi l'agente compilato
hierarchical_chunking_agent = create_hierarchical_chunking_agent()