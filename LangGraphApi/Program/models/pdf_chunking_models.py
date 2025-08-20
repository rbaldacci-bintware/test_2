# /Program/models/pdf_chunking_models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class Chunk(BaseModel):
    """Rappresenta un singolo chunk, di testo o tabellare."""
    content: Any = Field(..., description="Il contenuto del chunk, può essere stringa (testo) o dizionario (tabella).")
    metadata: Dict[str, Any] = Field(..., description="Metadati associati al chunk (es. pagina di origine).")

class PDFChunkingResponse(BaseModel):
    """Risposta finale dell'API di chunking orchestrato."""
    file_name: str = Field(..., description="Nome del file PDF elaborato.")
    total_chunks: int = Field(..., description="Numero totale di chunk generati.")
    message: str = Field(..., description="Messaggio di stato dell'operazione.")
    chunks: List[Chunk] = Field(..., description="La lista di tutti i chunk generati.")