# /Program/api/pdf_chunking_router.py
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException

# Importa i modelli e l'agente orchestratore
from models.pdf_chunking_models import PDFChunkingResponse
from agents.pdf_chunking_agent import orchestrator_agent

# Configura il logger
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/pdf-chunking", response_model=PDFChunkingResponse)
async def chunk_pdf_endpoint(file: UploadFile = File(...)):
    """
    Endpoint completo per analizzare, estrarre e creare chunk da un file PDF.
    Orchestra l'estrazione di testo e tabelle e il chunking semantico.
    """
    logger.info(f"Ricevuta richiesta di chunking completo per: {file.filename}")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="È richiesto un file PDF.")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Il file PDF è vuoto.")

        initial_state = {
            "pdf_file": contents,
            "pdf_name": file.filename,
        }
        
        config = {"configurable": {"thread_id": f"pdf_chunker_{uuid.uuid4()}"}}

        # Invoca l'agente orchestratore
        final_state = orchestrator_agent.invoke(initial_state, config=config)
        
        if final_state.get("error_message"):
            raise HTTPException(status_code=500, detail=final_state["error_message"])

        final_chunks = final_state.get("final_chunks", [])
        
        response = PDFChunkingResponse(
            file_name=file.filename,
            total_chunks=len(final_chunks),
            message=f"Elaborazione completata. Generati {len(final_chunks)} chunk.",
            chunks=final_chunks
        )
        
        return response

    except Exception as e:
        logger.error(f"Errore imprevisto nell'endpoint di chunking: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Errore interno del server: {str(e)}")