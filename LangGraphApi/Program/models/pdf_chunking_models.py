# /Program/models/pdf_chunking_models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class VectorDBFields(BaseModel):
    """Campi specifici per l'upsert nel vector database."""
    namespace: str = Field(..., description="Namespace/collection nel vector DB")
    score_boost: float = Field(..., description="Boost factor per il retrieval scoring")
    is_table: bool = Field(..., description="Flag per identificare chunk tabellari")
    page_number: int = Field(..., description="Numero di pagina per filtraggio")

class ChunkForUpsert(BaseModel):
    """Struttura di un chunk pronto per l'upsert in un vector database."""
    id: str = Field(..., description="ID univoco e deterministico per upsert idempotente")
    content: Any = Field(..., description="Contenuto del chunk (stringa per testo, dict per tabelle)")
    metadata: Dict[str, Any] = Field(..., description="Tutti i metadati del chunk")
    embedding_text: str = Field(..., description="Testo ottimizzato per generazione embeddings")
    text_representation: str = Field(..., description="Rappresentazione testuale per display/search")
    vector_db_fields: VectorDBFields = Field(..., description="Campi specifici per il vector DB")

class PDFChunkingResponse(BaseModel):
    """Risposta finale dell'API di chunking orchestrato."""
    file_name: str = Field(..., description="Nome del file PDF elaborato")
    total_chunks: int = Field(..., description="Numero totale di chunk generati")
    message: str = Field(..., description="Messaggio di stato dell'operazione")
    chunks: List[ChunkForUpsert] = Field(..., description="Lista di chunk pronti per upsert")
    
    class Config:
        schema_extra = {
            "example": {
                "file_name": "document.pdf",
                "total_chunks": 13,
                "message": "Elaborazione completata. Generati 13 chunk.",
                "chunks": [
                    {
                        "id": "document_a1b2c3d4e5f6",
                        "content": "Testo del chunk o dict per tabelle",
                        "metadata": {
                            "chunk_type": "text",
                            "source_page": 1,
                            "semantic_type": "introduction",
                            "key_concepts": ["concetto1", "concetto2"]
                        },
                        "embedding_text": "Testo ottimizzato per embeddings",
                        "text_representation": "Testo per display",
                        "vector_db_fields": {
                            "namespace": "document",
                            "score_boost": 0.8,
                            "is_table": False,
                            "page_number": 1
                        }
                    }
                ]
            }
        }