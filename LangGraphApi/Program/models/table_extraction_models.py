# /models/table_extraction_models.py
from pydantic import BaseModel, Field
from typing import List, Optional

class TableCell(BaseModel):
    """Rappresenta una singola cella della tabella"""
    rowIndex: int = Field(..., description="Indice della riga (base 1)")
    columnIndex: int = Field(..., description="Indice della colonna (base 1)")
    value: str = Field(..., description="Il contenuto testuale della cella")

class TableDimensions(BaseModel):
    """Dimensioni della tabella rilevata"""
    rows: int = Field(..., ge=0, description="Numero totale di righe")
    columns: int = Field(..., ge=0, description="Numero totale di colonne")

class ExtractedTableData(BaseModel):
    """Dati completi della tabella estratta"""
    table_cells: List[TableCell] = Field(default_factory=list, description="Lista delle celle estratte")

class ExtractionResponse(BaseModel):
    """Risposta finale dell'API di estrazione"""
    file_name: str = Field(..., description="Nome del file elaborato")
    rows: int = Field(..., ge=0, description="Numero di righe nella tabella")
    columns: int = Field(..., ge=0, description="Numero di colonne nella tabella")
    extracted_cells: List[TableCell] = Field(default_factory=list, description="Celle estratte dalla tabella")
    message: str = Field(..., description="Messaggio di stato dell'operazione")