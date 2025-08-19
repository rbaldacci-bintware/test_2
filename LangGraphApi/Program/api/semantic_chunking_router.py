# /api/semantic_chunking_router.py
from fastapi import APIRouter, HTTPException, Body
from pydantic import ValidationError
from models.semantic_chunking_models import (
    ChunkingRequest, 
    ChunkingResponse, 
    TableSchema,
    ChunkingStrategy
)
from models.table_extraction_models import TableCell
from agents.semantic_chunking_agent import chunking_agent
import uuid
import logging
from typing import Optional, Dict, Any

# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/semantic-chunking", response_model=ChunkingResponse)
async def create_semantic_chunks(request: ChunkingRequest):
    """
    Crea chunks semantici ottimizzati per RAG a partire da dati tabellari estratti.
    
    Questo endpoint:
    1. Analizza la struttura semantica della tabella
    2. Determina la strategia di chunking ottimale
    3. Genera chunks con metadati ricchi
    4. Valida la completezza dei chunks generati
    """
    logger.info(f"Richiesta chunking per file: {request.file_name}")
    logger.info(f"Dimensioni tabella: {request.rows}x{request.columns}")
    logger.info(f"Celle ricevute: {len(request.extracted_cells)}")
    
    # Validazione input
    if not request.extracted_cells:
        raise HTTPException(status_code=400, detail="Nessuna cella fornita per il chunking")
    
    if request.rows * request.columns != len(request.extracted_cells):
        logger.warning(f"Incoerenza dati: attese {request.rows * request.columns} celle, ricevute {len(request.extracted_cells)}")
    
    try:
        # Genera un ID univoco per il thread
        thread_id = str(uuid.uuid4())
        logger.info(f"Thread ID: {thread_id}")
        
        # Configurazione per l'agente
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 10
        }
        
        # Converti le celle in formato dizionario
        cells_dict = [cell.dict() for cell in request.extracted_cells]
        
        # Stato iniziale
        initial_state = {
            "file_name": request.file_name,
            "rows": request.rows,
            "columns": request.columns,
            "extracted_cells": cells_dict,
            "table_schema": None,
            "chunking_strategy": request.chunking_strategy.value if request.chunking_strategy else None,
            "chunks": [],
            "error_message": None
        }
        
        logger.info("Invocazione dell'agente di chunking semantico...")
        
        # Invoca l'agente
        final_state = chunking_agent.invoke(initial_state, config=config)
        
        logger.info(f"Agente completato. Chunks generati: {len(final_state.get('chunks', []))}")
        logger.info(f"Strategia utilizzata: {final_state.get('chunking_strategy')}")
        
        # Verifica errori
        if final_state.get("error_message"):
            logger.error(f"Errore nel chunking: {final_state['error_message']}")
            raise HTTPException(
                status_code=500,
                detail=f"Errore durante il chunking: {final_state['error_message']}"
            )
        
        # Verifica che ci siano chunks
        if not final_state.get("chunks"):
            logger.error("Nessun chunk generato")
            raise HTTPException(
                status_code=500,
                detail="Impossibile generare chunks dai dati forniti"
            )
        
        # Prepara la risposta
        response = ChunkingResponse(
            file_name=request.file_name,
            table_schema=TableSchema(**final_state.get("table_schema", {
                "headers": [],
                "table_type": "unknown",
                "primary_key_column": None,
                "semantic_structure": {}
            })),
            strategy_used=ChunkingStrategy(final_state.get("chunking_strategy", "row_based")),
            chunks=final_state.get("chunks", []),
            total_chunks=len(final_state.get("chunks", [])),
            message=f"Chunking completato con successo. Generati {len(final_state.get('chunks', []))} chunks usando strategia {final_state.get('chunking_strategy')}."
        )
        
        logger.info(f"Risposta preparata: {response.message}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Errore imprevisto nel chunking: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Errore interno del server: {str(e)}"
        )

@router.post("/semantic-chunking-from-json", response_model=ChunkingResponse)
async def create_semantic_chunks_from_json(data: Dict[str, Any]):
    """
    Endpoint alternativo che accetta dati JSON grezzi.
    Utile per test rapidi o integrazioni con sistemi esterni.
    
    Accetta lo stesso formato JSON dell'endpoint principale ma come dizionario generico.
    """
    try:
        logger.info(f"Ricevuto JSON con file: {data.get('file_name')}")
        
        # ** FIX: Pass the raw dictionary directly to the Pydantic model **
        # Pydantic will handle the validation and conversion of the nested TableCell objects.
        request = ChunkingRequest(**data)
        
        logger.info(f"ChunkingRequest creato con {len(request.extracted_cells)} celle")
        
        # Usa l'endpoint principale
        return await create_semantic_chunks(request)
        
    except ValidationError as e: # Catch Pydantic's validation error specifically
        logger.error(f"Errore nella validazione dei dati JSON: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Formato dati non valido: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Errore imprevisto nel chunking: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Errore interno del server: {str(e)}"
        )
