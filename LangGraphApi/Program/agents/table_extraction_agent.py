# /agents/table_extraction_agent.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RetryPolicy
from typing import TypedDict, List, Dict, Any, Literal, Optional
from services import gemini_service

# --- Stato del Grafico ---
class TableExtractionState(TypedDict):
    pdf_file_name: str
    pdf_base64: str
    pdf_mime_type: str
    rows: int
    columns: int
    extracted_cells: List[Dict[str, Any]]
    error_message: Optional[str]
    retry_count: int
    max_retries: int  # Aggiungiamo un limite massimo di tentativi

# --- Nodi del Grafico ---
def get_dimensions_node(state: TableExtractionState) -> dict:
    print(f"\n--- NODO: Get Dimensions (Tentativo {state.get('retry_count', 0) + 1}) ---")
    
    prompt = """Analizza il file PDF fornito. Trova la tabella principale nel documento. La tua unica risposta DEVE essere un oggetto JSON che descrive le sue dimensioni. Non includere testo aggiuntivo o markup.
    Schema JSON richiesto:
    {
        "rows": int,    // Il numero totale di righe nella tabella
        "columns": int  // Il numero totale di colonne nella tabella
    }"""
    
    try:
        response = gemini_service.call_gemini_api(
            prompt, 
            state["pdf_base64"], 
            state["pdf_mime_type"]
        )
        
        print(f"Tipo risposta dimensioni: {type(response)}")
        print(f"Risposta dimensioni: {response}")
        
        # Gestisci diversi formati di risposta
        if isinstance(response, dict):
            if "error" in response:
                print(f"Errore nel recupero dimensioni: {response['error']}")
                return {
                    "error_message": response["error"],
                    "rows": 0,
                    "columns": 0
                }
            
            rows = response.get("rows", 0)
            columns = response.get("columns", 0)
        elif isinstance(response, list) and len(response) > 0:
            # Se per qualche motivo ricevi una lista, prendi il primo elemento
            first_item = response[0]
            if isinstance(first_item, dict):
                rows = first_item.get("rows", 0)
                columns = first_item.get("columns", 0)
            else:
                print(f"Formato risposta lista non gestito: {response}")
                rows = 0
                columns = 0
        else:
            print(f"Formato risposta non riconosciuto: {type(response)}")
            rows = 0
            columns = 0
        
        print(f"Dimensioni rilevate: {rows} righe x {columns} colonne")
        
        # Verifica che le dimensioni siano valide
        if rows <= 0 or columns <= 0:
            return {
                "error_message": f"Dimensioni non valide: rows={rows}, columns={columns}",
                "rows": 0,
                "columns": 0
            }
        
        return {
            "rows": rows,
            "columns": columns,
            "error_message": None
        }
    except Exception as e:
        print(f"Eccezione in get_dimensions_node: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {
            "error_message": str(e),
            "rows": 0,
            "columns": 0
        }

def extract_data_node(state: TableExtractionState) -> dict:
    print(f"\n--- NODO: Extract Data ({state['rows']}x{state['columns']} celle) ---")
    
    # Prompt completo come nel codice C#
    prompt = f"""Analizza nuovamente lo stesso file PDF. Hai precedentemente identificato una tabella con {state['rows']} righe e {state['columns']} colonne.
    Ora estrai il contenuto di OGNI singola cella.
    La tua unica risposta DEVE essere un singolo oggetto JSON contenente una lista di tutte le celle.
    Schema JSON richiesto:
    {{
        "table_cells": [
            {{
                "rowIndex": int,      // Indice della riga (base 1)
                "columnIndex": int,   // Indice della colonna (base 1)
                "value": "string"     // Il contenuto testuale della cella
            }}
        ]
    }}
    Assicurati di restituire esattamente {state['rows'] * state['columns']} elementi nella lista 'table_cells'."""
    
    try:
        response = gemini_service.call_gemini_api(
            prompt, 
            state["pdf_base64"], 
            state["pdf_mime_type"]
        )
        
        print(f"Tipo di risposta: {type(response)}")
        print(f"Risposta estrazione (primi 500 caratteri): {str(response)[:500]}...")
        
        # Gestisci il caso in cui response sia già una lista (quando Gemini restituisce direttamente un array)
        if isinstance(response, list):
            print(f"Ricevuta lista diretta con {len(response)} elementi")
            # Se la risposta è direttamente una lista, assumiamo sia la lista delle celle
            table_cells = response
        elif isinstance(response, dict):
            # Se è un dizionario, cerca la chiave appropriata
            if "error" in response:
                print(f"Errore nell'estrazione dati: {response['error']}")
                return {
                    "error_message": response["error"],
                    "extracted_cells": []
                }
            
            # Prova diverse chiavi possibili
            table_cells = response.get("table_cells", 
                         response.get("tableCells", 
                         response.get("cells", 
                         response.get("data", []))))
            
            # Se non troviamo table_cells ma c'è una lista al root level, usala
            if not table_cells and isinstance(response, dict):
                # Cerca se c'è una chiave che contiene una lista
                for key, value in response.items():
                    if isinstance(value, list) and len(value) > 0:
                        print(f"Trovata lista sotto la chiave '{key}'")
                        table_cells = value
                        break
        else:
            print(f"Tipo di risposta non gestito: {type(response)}")
            table_cells = []
        
        # Converti il formato per essere compatibile con il modello Pydantic
        extracted_cells = []
        for cell in table_cells:
            if isinstance(cell, dict):
                extracted_cells.append({
                    "rowIndex": cell.get("rowIndex", cell.get("row", 0)),
                    "columnIndex": cell.get("columnIndex", cell.get("column", cell.get("col", 0))),
                    "value": str(cell.get("value", cell.get("text", cell.get("content", ""))))
                })
            else:
                print(f"Cella non è un dizionario: {cell}")
        
        print(f"Celle estratte: {len(extracted_cells)}")
        
        # Se non abbiamo estratto celle, segnala l'errore
        if len(extracted_cells) == 0:
            return {
                "error_message": f"Nessuna cella estratta. Formato risposta: {type(response)}",
                "extracted_cells": []
            }
        
        return {
            "extracted_cells": extracted_cells,
            "error_message": None
        }
    except Exception as e:
        print(f"Eccezione in extract_data_node: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {
            "error_message": str(e),
            "extracted_cells": []
        }

def validate_extraction_node(state: TableExtractionState) -> dict:
    print("\n--- NODO: Validate Extraction ---")
    
    expected = state['rows'] * state['columns']
    actual = len(state.get('extracted_cells', []))
    
    print(f"Validazione: attese {expected} celle, ricevute {actual}")
    
    # Incrementa il contatore dei tentativi
    new_retry_count = state.get('retry_count', 0) + 1
    
    if expected != actual:
        # Controlla se abbiamo superato il numero massimo di tentativi
        if new_retry_count >= state.get('max_retries', 3):
            error_msg = f"Raggiunto il numero massimo di tentativi ({state.get('max_retries', 3)}). Incoerenza persistente: attese {expected} celle, ricevute {actual}."
            print(f"ERRORE FINALE: {error_msg}")
            return {
                "error_message": error_msg,
                "retry_count": new_retry_count
            }
        else:
            error_msg = f"Incoerenza: attese {expected} celle, ricevute {actual}. Tentativo {new_retry_count}/{state.get('max_retries', 3)}"
            print(f"AVVISO: {error_msg}")
            return {
                "error_message": error_msg,
                "retry_count": new_retry_count
            }
    
    print("Validazione completata con successo!")
    return {
        "error_message": None,
        "retry_count": new_retry_count
    }

# --- Funzioni di Routing ---
def check_dimensions_validity(state: TableExtractionState) -> Literal["valid", "invalid"]:
    # Verifica se abbiamo un errore o dimensioni non valide
    if state.get("error_message") or not (state.get('rows', 0) > 0 and state.get('columns', 0) > 0):
        print(f"Dimensioni non valide: error_message={state.get('error_message')}, rows={state.get('rows', 0)}, columns={state.get('columns', 0)}")
        return "invalid"
    print("Dimensioni valide, procedo con l'estrazione")
    return "valid"

def check_extraction_consistency(state: TableExtractionState) -> Literal["consistent", "inconsistent", "max_retries_reached"]:
    # Se abbiamo raggiunto il numero massimo di tentativi, termina
    if state.get('retry_count', 0) >= state.get('max_retries', 3):
        print("Numero massimo di tentativi raggiunto, termino il processo")
        return "max_retries_reached"
    
    # Se c'è un errore ma non abbiamo raggiunto il limite, riprova
    if state.get("error_message"):
        print("Incoerenza rilevata, riprovo...")
        return "inconsistent"
    
    print("Estrazione consistente, processo completato")
    return "consistent"

# --- Costruzione e Compilazione del Grafico ---
def create_agent():
    retry_policy = RetryPolicy(max_attempts=3)
    checkpointer = InMemorySaver()

    workflow = StateGraph(TableExtractionState)
    
    # Aggiungi i nodi con retry policy
    workflow.add_node("get_dimensions_node", get_dimensions_node, retry_policy=retry_policy)
    workflow.add_node("extract_data_node", extract_data_node, retry_policy=retry_policy)
    workflow.add_node("validate_extraction_node", validate_extraction_node)

    # Definisci il flusso
    workflow.set_entry_point("get_dimensions_node")
    
    workflow.add_conditional_edges(
        "get_dimensions_node", 
        check_dimensions_validity, 
        {
            "valid": "extract_data_node", 
            "invalid": END
        }
    )
    
    workflow.add_edge("extract_data_node", "validate_extraction_node")
    
    workflow.add_conditional_edges(
        "validate_extraction_node", 
        check_extraction_consistency, 
        {
            "consistent": END, 
            "inconsistent": "get_dimensions_node",
            "max_retries_reached": END
        }
    )

    return workflow.compile(checkpointer=checkpointer)

# Esponiamo un'unica istanza compilata dell'agente
extraction_agent = create_agent()