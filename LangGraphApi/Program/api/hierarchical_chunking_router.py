# /Program/api/hierarchical_chunking_router.py
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# Importa l'agente gerarchico
from agents.hierarchical_chunking_agent import hierarchical_chunking_agent

# Configura il logger
logger = logging.getLogger(__name__)

router = APIRouter()

# Modelli per la risposta
class HierarchyInfo(BaseModel):
    """Informazioni sulla gerarchia di un chunk."""
    title: str = Field(None, description="Titolo del documento")
    chapter: str = Field(None, description="Capitolo di appartenenza")
    paragraph: str = Field(None, description="Paragrafo di appartenenza")
    subparagraph: str = Field(None, description="Sottoparagrafo di appartenenza")
    subsubparagraph: str = Field(None, description="Sotto-sottoparagrafo di appartenenza")

class HierarchicalChunk(BaseModel):
    """Chunk con informazioni gerarchiche complete."""
    id: str = Field(..., description="ID univoco del chunk")
    content: Any = Field(..., description="Contenuto del chunk")
    metadata: Dict[str, Any] = Field(..., description="Metadati completi del chunk")
    hierarchy: HierarchyInfo = Field(..., description="Informazioni gerarchiche")
    embedding_text: str = Field(..., description="Testo ottimizzato per embeddings")
    text_representation: str = Field(..., description="Rappresentazione testuale")
    vector_db_fields: Dict[str, Any] = Field(..., description="Campi per vector DB")

class DocumentStructure(BaseModel):
    """Struttura gerarchica del documento."""
    document_title: str = Field(..., description="Titolo del documento")
    structure_type: str = Field(..., description="Tipo di struttura (numbered/named/mixed)")
    hierarchy_patterns: Dict[str, str] = Field(..., description="Pattern identificati per ogni livello")
    identified_sections: List[Dict[str, Any]] = Field(..., description="Sezioni identificate")
    structure_confidence: float = Field(..., description="Confidenza nell'analisi strutturale")

class HierarchicalChunkingResponse(BaseModel):
    """Risposta dell'API di chunking gerarchico."""
    file_name: str = Field(..., description="Nome del file PDF elaborato")
    document_structure: DocumentStructure = Field(..., description="Struttura gerarchica identificata")
    total_chunks: int = Field(..., description="Numero totale di chunk generati")
    text_chunks_count: int = Field(..., description="Numero di chunk di testo")
    table_chunks_count: int = Field(..., description="Numero di chunk di tabelle")
    special_chunks_count: int = Field(..., description="Numero di chunk speciali")
    hierarchy_elements_count: int = Field(..., description="Numero di elementi gerarchici identificati")
    chunks: List[HierarchicalChunk] = Field(..., description="Lista di chunk con gerarchia")
    message: str = Field(..., description="Messaggio di stato dell'operazione")

@router.post("/hierarchical-chunking", response_model=HierarchicalChunkingResponse)
async def hierarchical_chunk_pdf_endpoint(file: UploadFile = File(...)):
    """
    Endpoint per chunking gerarchico avanzato di PDF.
    
    Questo endpoint:
    1. Analizza la struttura gerarchica del documento (titoli, capitoli, paragrafi, sottoparagrafi)
    2. Marca univocamente ogni elemento per facilitare la ricerca
    3. Estrae testo e tabelle mantenendo il contesto gerarchico
    4. Crea chunk semantici con metadati gerarchici completi
    5. Gestisce casi speciali con l'aiuto di LLM
    
    Returns:
        HierarchicalChunkingResponse con tutti i chunk e la struttura del documento
    """
    logger.info(f"Ricevuta richiesta di chunking gerarchico per: {file.filename}")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="È richiesto un file PDF.")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Il file PDF è vuoto.")

        # Stato iniziale per l'agente
        initial_state = {
            "pdf_file": contents,
            "pdf_name": file.filename,
            "pdf_metadata": {},
            "document_structure": {},
            "hierarchy_elements": [],
            "text_blocks": [],
            "tables_info": [],
            "text_chunks": [],
            "table_chunks": [],
            "final_chunks": [],
            "error_message": None,
            "special_cases": []
        }
        
        # Configurazione per l'agente
        config = {"configurable": {"thread_id": f"hierarchical_{uuid.uuid4()}"}}

        logger.info("Invocazione dell'agente di chunking gerarchico...")
        
        # Invoca l'agente gerarchico
        final_state = hierarchical_chunking_agent.invoke(initial_state, config=config)
        
        # Verifica errori
        if final_state.get("error_message"):
            logger.error(f"Errore nel chunking gerarchico: {final_state['error_message']}")
            raise HTTPException(status_code=500, detail=final_state["error_message"])

        # Estrai i risultati
        final_chunks = final_state.get("final_chunks", [])
        document_structure = final_state.get("document_structure", {})
        hierarchy_elements = final_state.get("hierarchy_elements", [])
        
        # Conta i tipi di chunk
        text_count = len([c for c in final_chunks if c["metadata"].get("chunk_type") == "text"])
        table_count = len([c for c in final_chunks if c["metadata"].get("chunk_type") == "table"])
        special_count = len([c for c in final_chunks if c["metadata"].get("chunk_type") == "text_special"])
        
        # Prepara i chunk per la risposta
        formatted_chunks = []
        for chunk in final_chunks:
            formatted_chunk = HierarchicalChunk(
                id=chunk["id"],
                content=chunk["content"],
                metadata=chunk["metadata"],
                hierarchy=HierarchyInfo(**chunk.get("hierarchy", {})),
                embedding_text=chunk["embedding_text"],
                text_representation=chunk["text_representation"],
                vector_db_fields=chunk["vector_db_fields"]
            )
            formatted_chunks.append(formatted_chunk)
        
        # Prepara la struttura del documento
        doc_structure = DocumentStructure(
            document_title=document_structure.get("document_title", file.filename),
            structure_type=document_structure.get("structure_type", "unknown"),
            hierarchy_patterns=document_structure.get("hierarchy_patterns", {}),
            identified_sections=document_structure.get("identified_sections", []),
            structure_confidence=document_structure.get("structure_confidence", 0.0)
        )
        
        response = HierarchicalChunkingResponse(
            file_name=file.filename,
            document_structure=doc_structure,
            total_chunks=len(final_chunks),
            text_chunks_count=text_count,
            table_chunks_count=table_count,
            special_chunks_count=special_count,
            hierarchy_elements_count=len(hierarchy_elements),
            chunks=formatted_chunks,
            message=f"Chunking gerarchico completato. Identificati {len(hierarchy_elements)} elementi gerarchici, generati {len(final_chunks)} chunk totali."
        )
        
        logger.info(f"Risposta preparata: {response.message}")
        logger.info(f"Struttura identificata: {doc_structure.structure_type} con confidenza {doc_structure.structure_confidence}")
        
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Errore imprevisto nel chunking gerarchico: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Errore interno del server: {str(e)}")

@router.get("/hierarchical-chunking/test")
async def test_hierarchical_chunking():
    """Endpoint di test per verificare che il router sia caricato correttamente."""
    return {
        "status": "ok",
        "message": "Hierarchical Chunking API è attiva",
        "description": "Usa POST /hierarchical-chunking con un file PDF per elaborarlo"
    }