# /api/table_extraction_router.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from models.table_extraction_models import ExtractionResponse
from agents.table_extraction_agent import extraction_agent
import base64
import uuid
import logging

# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/extract-table", response_model=ExtractionResponse)
async def extract_table_from_pdf(file: UploadFile = File(...)):
    """
    Estrae una tabella da un file PDF in due step: 
    prima ne calcola le dimensioni, poi ne estrae i dati.
    """
    logger.info(f"Ricevuto file: {file.filename}, Content-Type: {file.content_type}")
    
    # Validazione del file
    if file.content_type != "application/pdf":
        logger.warning(f"Tipo di file non valido: {file.content_type}")
        raise HTTPException(status_code=400, detail="È richiesto un file PDF valido.")
    
    try:
        # Leggi il contenuto del file
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Il file PDF è vuoto.")
        
        logger.info(f"File letto correttamente, dimensione: {len(contents)} bytes")
        
        # Converti in base64
        base64_pdf = base64.b64encode(contents).decode('utf-8')
        
        # Genera un ID univoco per il thread
        thread_id = str(uuid.uuid4())
        logger.info(f"Thread ID generato: {thread_id}")
        
        # Configurazione per l'agente
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 10  # Aumentato per permettere più tentativi
        }
        
        # Stato iniziale con tutti i campi necessari
        initial_state = {
            "pdf_file_name": file.filename,
            "pdf_base64": base64_pdf,
            "pdf_mime_type": file.content_type,
            "retry_count": 0,
            "max_retries": 8,  # Numero massimo di tentativi
            "rows": 0,
            "columns": 0,
            "extracted_cells": [],
            "error_message": None
        }
        
        logger.info("Invocazione dell'agente di estrazione...")
        
        # Invoca l'agente
        final_state = extraction_agent.invoke(initial_state, config=config)
        
        logger.info(f"Agente completato. Stato finale: rows={final_state.get('rows', 0)}, columns={final_state.get('columns', 0)}, cells={len(final_state.get('extracted_cells', []))}")
        
        # Verifica se c'è stato un errore
        if final_state.get("error_message"):
            logger.error(f"Errore nell'estrazione: {final_state['error_message']}")
            raise HTTPException(
                status_code=500, 
                detail=f"Errore durante l'estrazione: {final_state['error_message']}"
            )
        
        # Verifica che ci siano dati estratti
        if not final_state.get("extracted_cells"):
            logger.warning("Nessuna cella estratta dal documento")
            raise HTTPException(
                status_code=404, 
                detail="Impossibile estrarre la tabella dal documento fornito."
            )
        
        # Costruisci la risposta
        response = ExtractionResponse(
            file_name=final_state["pdf_file_name"],
            rows=final_state.get("rows", 0),
            columns=final_state.get("columns", 0),
            extracted_cells=final_state.get("extracted_cells", []),
            message=f"Estrazione completata con successo. Estratte {len(final_state.get('extracted_cells', []))} celle."
        )
        
        logger.info(f"Risposta preparata con successo: {response.message}")
        return response
        
    except HTTPException:
        # Rilancia le HTTPException già gestite
        raise
    except Exception as e:
        # Gestisci eventuali errori imprevisti
        logger.error(f"Errore imprevisto durante l'estrazione della tabella: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Si è verificato un errore interno del server: {str(e)}"
        )