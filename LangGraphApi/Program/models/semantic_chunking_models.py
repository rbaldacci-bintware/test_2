# /models/semantic_chunking_models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from enum import Enum

class ChunkingStrategy(str, Enum):
    """Strategie di chunking disponibili"""
    ROW_BASED = "row_based"
    COLUMN_BASED = "column_based"
    HYBRID = "hybrid"

class TableCell(BaseModel):
    """Rappresenta una singola cella della tabella"""
    rowIndex: int = Field(..., description="Indice della riga (base 1)")
    columnIndex: int = Field(..., description="Indice della colonna (base 1)")
    value: str = Field(..., description="Il contenuto testuale della cella")

class TableSchema(BaseModel):
    """Schema semantico della tabella analizzata"""
    headers: List[str] = Field(..., description="Header identificati nella tabella")
    table_type: str = Field(..., description="Tipo di tabella (azioni, comparativa, statistica, etc.)")
    primary_key_column: Optional[int] = Field(None, description="Colonna che funge da chiave primaria")
    semantic_structure: Dict[str, Any] = Field(..., description="Struttura semantica della tabella")

class ChunkMetadata(BaseModel):
    """Metadati per ogni chunk"""
    chunk_id: str = Field(..., description="ID univoco del chunk")
    source_file: str = Field(..., description="File di origine")
    chunk_type: str = Field(..., description="Tipo di chunk")
    row_indices: Optional[List[int]] = Field(None, description="Indici delle righe incluse")
    column_indices: Optional[List[int]] = Field(None, description="Indici delle colonne incluse")
    priority: Optional[str] = Field(None, description="Livello di priorità se applicabile")
    keywords: List[str] = Field(default_factory=list, description="Parole chiave estratte")

class SemanticChunk(BaseModel):
    """Rappresenta un chunk semantico per RAG"""
    metadata: ChunkMetadata = Field(..., description="Metadati del chunk")
    content: Dict[str, Any] = Field(..., description="Contenuto strutturato del chunk")
    text_representation: str = Field(..., description="Rappresentazione testuale per embedding")
    embedding_text: str = Field(..., description="Testo ottimizzato per generazione embedding")

class ChunkingRequest(BaseModel):
    """Richiesta di chunking semantico"""
    file_name: str = Field(..., description="Nome del file sorgente")
    rows: int = Field(..., description="Numero di righe nella tabella")
    columns: int = Field(..., description="Numero di colonne nella tabella")
    extracted_cells: List[TableCell] = Field(..., description="Celle estratte dalla tabella")
    chunking_strategy: Optional[ChunkingStrategy] = Field(None, description="Strategia di chunking (opzionale, auto-detect se non specificata)")

class ChunkingResponse(BaseModel):
    """Risposta del servizio di chunking semantico"""
    file_name: str = Field(..., description="Nome del file elaborato")
    table_schema: TableSchema = Field(..., description="Schema semantico identificato")
    strategy_used: ChunkingStrategy = Field(..., description="Strategia di chunking utilizzata")
    chunks: List[SemanticChunk] = Field(..., description="Chunks generati")
    total_chunks: int = Field(..., description="Numero totale di chunks generati")
    message: str = Field(..., description="Messaggio di stato dell'operazione")